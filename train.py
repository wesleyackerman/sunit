"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument('--trainer', type=str, default='SECUNIT', help="CDUNIT|SECUNIT|MUNIT|UNIT")
opts = parser.parse_args()

if opts.trainer == "SECUNIT":
    from secunit_utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, __write_images
elif opts.trainer == "CDUNIT":
    from cd_utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, __write_images
else:
    from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, __write_images

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
opts.output_path = config['vgg_model_path']

# Setup model and data loader
if opts.trainer == 'MUNIT':
    from trainer import MUNIT_Trainer
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    from trainer import UNIT_Trainer
    trainer = UNIT_Trainer(config)
elif opts.trainer == 'CDUNIT':
    from cd_trainer import CD_TRAINER
    trainer = CDUNIT_Trainer(config)
elif opts.trainer == 'SECUNIT':
    from secunit_trainer import SECUNIT_Trainer
    trainer = SECUNIT_Trainer(config)
else:
    sys.exit("Only support CDUNIT|SECUNIT|MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config)

if opts.trainer == "SECUNIT":
    train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda())
    train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda())
    train_display_seg_a = Variable(torch.stack([train_loader_a.dataset[i][1] for i in range(display_size)]).cuda())
    train_display_seg_b = Variable(torch.stack([train_loader_b.dataset[i][1] for i in range(display_size)]).cuda())
    train_display_seg_a = (train_display_seg_a * 255).type(torch.cuda.LongTensor)
    train_display_seg_b = (train_display_seg_b * 255).type(torch.cuda.LongTensor)

    test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i][0] for i in range(display_size)]).cuda())
    test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i][0] for i in range(display_size)]).cuda())
    test_display_seg_a = Variable(torch.stack([test_loader_a.dataset[i][1] for i in range(display_size)]).cuda())
    test_display_seg_b = Variable(torch.stack([test_loader_b.dataset[i][1] for i in range(display_size)]).cuda())
    test_display_seg_a = (test_display_seg_a * 255).type(torch.cuda.LongTensor)
    test_display_seg_b = (test_display_seg_b * 255).type(torch.cuda.LongTensor)

elif opts.trainer == "CDUNIT":
    train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i][0] for i in range(display_size)]).cuda())
    train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i][0] for i in range(display_size)]).cuda())
    test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda())
    test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda())

else:
    train_display_images_a = Variable(torch.stack([train_loader_a.dataset[i] for i in range(display_size)]).cuda())
    train_display_images_b = Variable(torch.stack([train_loader_b.dataset[i] for i in range(display_size)]).cuda())
    test_display_images_a = Variable(torch.stack([test_loader_a.dataset[i] for i in range(display_size)]).cuda())
    test_display_images_b = Variable(torch.stack([test_loader_b.dataset[i] for i in range(display_size)]).cuda())

# Setup logger and output folders
model_name = config['exp_name'] #os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
if opts.trainer == 'SECUNIT':
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_a_seg, images_b, images_b_seg  = Variable(images_a[0].cuda()), Variable(images_a[1].cuda()), \
                                                                Variable(images_b[0].cuda()), Variable(images_b[1].cuda())
            
            images_a_seg = (images_a_seg * 255).type(torch.cuda.LongTensor)
            images_b_seg = (images_b_seg * 255).type(torch.cuda.LongTensor)

            # Main training code
            trainer.dis_update(images_a, images_a_seg, images_b, images_b_seg, config)
            trainer.gen_update(images_a, images_a_seg, images_b, images_b_seg, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs, test_seg = trainer.sample(test_display_images_a, test_display_seg_a, test_display_images_b, test_display_seg_b)
                    train_image_outputs, train_seg = trainer.sample(train_display_images_a, train_display_seg_a, train_display_images_b, train_display_seg_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                #write_2images(test_seg, display_size, image_directory, 'seg_test_%08d' % (iterations + 1))
                #write_2images(train_seg, display_size, image_directory, 'seg_train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs, image_seg = trainer.sample(train_display_images_a, train_display_seg_a, train_display_images_b, train_display_seg_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')
                seg_mask = []
                for i in range(len(image_seg)):
                    seg_mask.append(torch.max(image_seg[i], 1, keepdim=True)[1])
                write_2images(seg_mask, display_size, image_directory, 'seg_train_current', norm=False)

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

elif opts.trainer == 'CDUNIT':
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_a_seg, images_b, images_b_seg  = Variable(images_a[0].cuda()), Variable(images_a[1].cuda()), \
                                                                Variable(images_b[0].cuda()), Variable(images_b[1].cuda())

            images_a_seg = (images_a_seg * 255).type(torch.cuda.LongTensor)
            images_b_seg = (images_b_seg * 255).type(torch.cuda.LongTensor)

            # Main training code
            trainer.dis_update(images_a, images_a_seg, images_b, images_b_seg, config)
            trainer.gen_update(images_a, images_a_seg, images_b, images_b_seg, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs, test_seg = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs, train_seg = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                #write_2images(test_seg, display_size, image_directory, 'seg_test_%08d' % (iterations + 1))
                #write_2images(train_seg, display_size, image_directory, 'seg_train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs, image_seg = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')
                seg_mask = []
                for i in range(len(image_seg)):
                    seg_mask.append(torch.max(image_seg[i], 1, keepdim=True)[1])
                write_2images(seg_mask, display_size, image_directory, 'seg_train_current', norm=False)

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')

else:
    while True:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            trainer.update_learning_rate()
            images_a, images_b = Variable(images_a.cuda()), Variable(images_b.cuda())

            # Main training code
            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

            # Write images
            if (iterations + 1) % config['image_save_iter'] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                    train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
                write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
                # HTML
                write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

            if (iterations + 1) % config['image_display_iter'] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
                write_2images(image_outputs, display_size, image_directory, 'train_current')

            # Save network weights
            if (iterations + 1) % config['snapshot_save_iter'] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                sys.exit('Finish training')


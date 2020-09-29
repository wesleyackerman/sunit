"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--input', type=str, help="input image path")
parser.add_argument('--output_folder', type=str, help="output image path")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='SECUNIT', help="SECUNIT|CDUNIT|MUNIT|UNIT")

CTSCP_PALETTE = [0, 0, 0,
        128, 64, 128,      # road
        244, 35, 232,      # sidewalk
        70, 70, 70,       # building
        102, 102, 156,    # wall
        190, 153, 153,    # fence
        153, 153, 153,    # pole
        250, 170, 30,     # traffic light
        220, 220,  0,     # traffic sign
        107, 142, 35,     # vegetation
        152, 251, 152,    # terrain
        0, 130, 180,      # sky
        220, 20, 60,      # person
        255, 0, 0,        # rider
        0, 0, 142,        # car
        0, 0, 70,         # truck
        0, 60, 100,       # bus
        0, 80, 100,       # train
        0, 0, 230,        # motorcycle
        119, 11, 32,
        250, 250, 250,
        250, 250, 0,
        250, 0, 250,
        140, 140, 140,
        70, 0, 0,
        0, 100, 0]

def write_seg(tensor, file_name):
    if len(tensor.size()) == 4:
        tensor = tensor.squeeze(0)
    elif len(tensor.size()) == 2:
        tensor = tensor.unsqueeze(0)
    image = tensor.permute(1, 2, 0).to('cpu', torch.uint8).numpy()[:,:,0] # 0 index to select single channel, all channels are duplicates
    image = Image.fromarray(image.squeeze(), mode="L")
    image.putpalette(CTSCP_PALETTE)
    image.save(file_name)

def write_img_set(imgs, style_imgs, fake_seg_imgs, real_seg_imgs, orig_dir, tran_dir, seg_dir, style_dir, count):
    orig_imgs = imgs[0]
    if fake_seg_imgs is not None:
        fake_seg_imgs = torch.max(fake_seg_imgs, 1, keepdim=True)[1]

    for i in range(orig_imgs.size(0)):
        vutils.save_image(orig_imgs[i], '%s/%d.png' % (orig_dir, count + i), normalize=True)
        if fake_seg_imgs is not None:
            write_seg(fake_seg_imgs[i], '%s/%d.png' % (Path(seg_dir, "tran"), count + i))
            write_seg(real_seg_imgs[i], '%s/%d.png' % (Path(seg_dir, "orig"), count + i))
            #vutils.save_image(seg_imgs[i], '%s/%d.png' % (seg_dir, count + i))
        for j in range(2, len(imgs)):
            vutils.save_image(imgs[j][i], '%s/%d_%d.png' % (tran_dir, count + i, j-1), normalize=True)

        for j in range(len(style_imgs)):
            vutils.save_image(style_imgs[j][i], '%s/%d_%d.png' % (style_dir, count + i, j+1), normalize=True)

    return count + orig_imgs.size(0)

def enum_loader(loader_a, loader_b, orig_dir, tran_dir, seg_dir, style_dir, opts, img_count_a, img_count_b, n_rand):
    Path(orig_dir, "A").mkdir(exist_ok=True)
    Path(tran_dir, "A").mkdir(exist_ok=True)
    Path(seg_dir, "A", "orig").mkdir(parents=True, exist_ok=True)
    Path(seg_dir, "A", "tran").mkdir(exist_ok=True)
    Path(orig_dir, "B").mkdir(exist_ok=True)
    Path(tran_dir, "B").mkdir(exist_ok=True)
    Path(seg_dir, "B", "orig").mkdir(parents=True, exist_ok=True)
    Path(seg_dir, "B", "tran").mkdir(exist_ok=True)
    Path(style_dir, "A").mkdir(exist_ok=True)
    Path(style_dir, "B").mkdir(exist_ok=True)

    for images_a in loader_a:
        images_a, images_a_seg = Variable(images_a[0].cuda()), Variable(images_a[1].cuda())
        images_a_seg = (images_a_seg * 255).type(torch.cuda.LongTensor)

        if opts.trainer == 'SECUNIT':
            a_sampled, ab_sampled_seg, style_samples = trainer.sample_a(images_a, images_a_seg, n_random_codes=n_rand)
        else:
            a_sampled, ab_sampled_seg, style_samples = trainer.sample_a(images_a, n_random_codes=n_rand)
        
        img_count_a = write_img_set(a_sampled, style_samples, ab_sampled_seg, images_a_seg, Path(orig_dir, "A"), Path(tran_dir, "B"), Path(seg_dir, "A"), Path(style_dir, "A"), img_count_a)

    for images_b in loader_b:
        images_b, images_b_seg = Variable(images_b[0].cuda()), Variable(images_b[1].cuda())
        images_b_seg = (images_b_seg * 255).type(torch.cuda.LongTensor)

        if opts.trainer == 'SECUNIT':
            b_sampled, ba_sampled_seg, style_samples = trainer.sample_b(images_b, images_b_seg, n_random_codes=n_rand)
        else:
            b_sampled, ba_sampled_seg, style_samples = trainer.sample_b(images_b, n_random_codes=n_rand)
            
        img_count_b = write_img_set(b_sampled, style_samples, ba_sampled_seg, images_b_seg, Path(orig_dir, "B"), Path(tran_dir, "A"), Path(seg_dir, "B"), Path(style_dir, "B"), img_count_b)

    return img_count_a, img_count_b

def enum_munit_loader(loader_a, loader_b, orig_dir, tran_dir, style_dir, opts, img_count_a, img_count_b, n_rand):
    Path(orig_dir, "A").mkdir(exist_ok=True)
    Path(tran_dir, "A").mkdir(exist_ok=True)
    Path(orig_dir, "B").mkdir(exist_ok=True)
    Path(tran_dir, "B").mkdir(exist_ok=True)
    Path(style_dir, "A").mkdir(exist_ok=True)
    Path(style_dir, "B").mkdir(exist_ok=True)

    for images_a in loader_a:
        images_a = Variable(images_a.cuda())
        a_sampled, style_samples = trainer.sample_a(images_a, n_random_codes=n_rand)
        img_count_a = write_img_set(a_sampled, style_samples, None, None, Path(orig_dir, "A"), Path(tran_dir, "B"), None, Path(style_dir, "A"), img_count_a)
        
    for images_b in loader_b:
        images_b = Variable(images_b.cuda())
        b_sampled, style_samples = trainer.sample_b(images_b, n_random_codes=n_rand)
        img_count_b = write_img_set(b_sampled, style_samples, None, None, Path(orig_dir, "B"), Path(tran_dir, "A"), None, Path(style_dir, "B"), img_count_b)

    return img_count_a, img_count_b


opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
opts.num_style = 1 if opts.style != '' else opts.num_style

if opts.trainer == 'MUNIT':
    from trainer import MUNIT_Trainer
    from utils import get_all_data_loaders, get_config
    config = get_config(opts.config)
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    from trainer import UNIT_Trainer
    from utils import get_config
    config = get_config(opts.config)
    trainer = UNIT_Trainer(config)
elif opts.trainer == 'CDUNIT':
    from cd_trainer import CDUNIT_Trainer
    from cd_utils import get_all_data_loaders, get_config
    config = get_config(opts.config)
    trainer = CDUNIT_Trainer(config)
elif opts.trainer == 'SECUNIT':
    from secunit_trainer import SECUNIT_Trainer
    from secunit_utils import get_all_data_loaders, get_config
    config = get_config(opts.config)
    trainer = SECUNIT_Trainer(config)
else:
    sys.exit("Only support SECUNIT|CDUNIT|MUNIT|UNIT")

config['vgg_model_path'] = opts.output_path
style_dim = config['gen']['style_dim']

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

start = time.time()
n_rand = 3


if opts.trainer == 'SECUNIT' or opts.trainer == 'CDUNIT':
    trainer.resume(opts.checkpoint, hyperparameters=config)
    trainer.cuda()
    trainer.eval()
    config['batch_size'] = 8
    img_count_a, img_count_b = 0, 0
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config, train_bool=False)
    
    base_dir = Path(opts.output_folder)
    orig_dir = Path(base_dir, "orig")
    orig_dir.mkdir(parents=True, exist_ok=True)
    tran_dir = Path(base_dir, "fake")
    tran_dir.mkdir(exist_ok=True)
    seg_dir = Path(base_dir, "seg")
    seg_dir.mkdir(exist_ok=True)
    style_dir = Path(base_dir, "style")
    style_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        img_count_a, img_count_b = enum_loader(train_loader_a, train_loader_b, orig_dir, tran_dir, seg_dir, style_dir, opts, img_count_a, img_count_b, n_rand)
        img_count_a, img_count_b = enum_loader(test_loader_a, test_loader_b, orig_dir, tran_dir, seg_dir, style_dir, opts, img_count_a, img_count_b, n_rand)
    
    print(str(img_count_a) + " images used from domain A")
    print(str(img_count_b) + " images used from domain B")
    end = time.time()
    print("Runtime in seconds: ")
    print(end - start)
    sys.exit()

if opts.trainer == 'MUNIT':
    trainer.resume(opts.checkpoint, hyperparameters=config)
    trainer.cuda()
    trainer.eval()
    config['batch_size'] = 8
    img_count_a, img_count_b = 0, 0
    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_all_data_loaders(config, train_bool=False)

    base_dir = Path(opts.output_folder)
    orig_dir = Path(base_dir, "orig")
    orig_dir.mkdir(parents=True, exist_ok=True)
    tran_dir = Path(base_dir, "fake")
    tran_dir.mkdir(exist_ok=True)
    style_dir = Path(base_dir, "style")
    style_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        #img_count_a, img_count_b = enum_munit_loader(train_loader_a, train_loader_b, orig_dir, tran_dir, style_dir, opts, img_count_a, img_count_b, n_rand)
        img_count_a, img_count_b = enum_munit_loader(test_loader_a, test_loader_b, orig_dir, tran_dir, style_dir, opts, img_count_a, img_count_b, n_rand)

    print(str(img_count_a) + " images used from domain A")
    print(str(img_count_b) + " images used from domain B")
    end = time.time()
    print("Runtime in seconds: ")
    print(end - start)
    sys.exit()

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

with torch.no_grad():
    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    image = Variable(transform(Image.open(opts.input).convert('RGB')).unsqueeze(0).cuda())
    style_image = Variable(transform(Image.open(opts.style).convert('RGB')).unsqueeze(0).cuda()) if opts.style != '' else None

    # Start testing
    content, _ = encode(image)

    if opts.trainer == 'MUNIT':
        style_rand = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
        if opts.style != '':
            _, style = style_encode(style_image)
        else:
            style = style_rand
        for j in range(opts.num_style):
            s = style[j].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            path = os.path.join(opts.output_folder, 'output{:03d}.jpg'.format(j))
            vutils.save_image(outputs.data, path, padding=0, normalize=True)
    elif opts.trainer == 'UNIT':
        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        path = os.path.join(opts.output_folder, 'output.jpg')
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
    else:
        pass

    if not opts.output_only:
        # also save input images
        vutils.save_image(image.data, os.path.join(opts.output_folder, 'input.jpg'), padding=0, normalize=True)


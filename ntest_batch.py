"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
from trainer import MUNIT_Trainer, UNIT_Trainer
from cd_trainer import CDUNIT_Trainer
from torch import nn
from scipy.stats import entropy
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from data import ImageFolder
import numpy as np
import torchvision.utils as vutils
from PerceptualSimilarity import PerceptualLoss
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import sys
import torch
import os


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder', help='Path to the config file.')
parser.add_argument('--input_folder', type=str, help="input image folder")
parser.add_argument('--output_folder', type=str, help="output image folder")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, help="1 for a2b and others for b2a", default=1)
parser.add_argument('--seed', type=int, default=1, help="random seed")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--synchronized', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_only', action='store_true', help="whether use synchronized style code or not")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--trainer', type=str, default='CDUNIT', help="CDUNIT|MUNIT|UNIT")
parser.add_argument('--compute_IS', action='store_true', help="whether to compute Inception Score or not")
parser.add_argument('--compute_CIS', action='store_true', help="whether to compute Conditional Inception Score or not")
parser.add_argument('--compute_LPIPS', action='store_true', help="whether to compute LPIPS distances or not")
parser.add_argument('--inception_a', type=str, default='.', help="path to the pretrained inception network for domain A")
parser.add_argument('--inception_b', type=str, default='.', help="path to the pretrained inception network for domain B")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)

if opts.trainer == 'MUNIT':
    from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
elif opts.trainer == 'UNIT':
    from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
elif opts.trainer == 'CDUNIT':
    from cd_utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
else:
    sys.exit("Only support CDUNIT|MUNIT|UNIT")

# Load experiment setting
config = get_config(opts.config)
input_dim = config['input_dim_a'] if opts.a2b else config['input_dim_b']

# Load the inception networks if we need to compute IS or CIIS
if opts.compute_IS or opts.compute_IS:
    inception = load_inception(opts.inception_b) if opts.a2b else load_inception(opts.inception_a)
    inception.cuda()
    # freeze the inception models and set eval mode
    inception.eval()
    for param in inception.parameters():
        param.requires_grad = False
    inception_up = nn.Upsample(size=(299, 299), mode='bilinear')

# Setup model and data loader
image_names = ImageFolder(opts.input_folder, transform=None, return_paths=True)
data_loader = get_data_loader_folder(opts.input_folder, 1, True, new_size=config['new_size'], crop=False) # Shuffle False to True

config['vgg_model_path'] = opts.output_path
if opts.trainer == 'MUNIT':
    style_dim = config['gen']['style_dim']
    trainer = MUNIT_Trainer(config)
    from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
    from utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
elif opts.trainer == 'CDUNIT':
    style_dim = config['gen']['style_dim']
    trainer = CDUNIT_Trainer(config)
    from cd_utils import get_config, get_data_loader_folder, pytorch03_to_pytorch04, load_inception
else:
    sys.exit("Only support CDUNIT|MUNIT|UNIT")

try:
    state_dict = torch.load(opts.checkpoint)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])
except:
    state_dict = pytorch03_to_pytorch04(torch.load(opts.checkpoint), opts.trainer)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if opts.compute_IS:
    IS = []
    all_preds = []
if opts.compute_CIS:
    CIS = []
if opts.compute_LPIPS:
    LPIPS = []
    dist_model = PerceptualLoss(net='alex')

if opts.trainer == 'CDUNIT':
    with torch.no_grad():
        # Start testing
        style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            #if opts.compute_LPIPS and i > 200:
            #    break
            if opts.compute_CIS:
                cur_preds = []
            print(names[1])
            images = Variable(images.cuda())
            content, _, content_seg = encode(images)
            style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda())
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s, content_seg)
                outputs = (outputs + 1) / 2.
                if opts.compute_IS or opts.compute_CIS:
                    pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
                if opts.compute_IS:
                    all_preds.append(pred)
                if opts.compute_CIS:
                    cur_preds.append(pred)
                if opts.compute_LPIPS:
                    rand_code = torch.randn(1, style_dim, 1, 1).cuda()
                    output2 = decode(content, rand_code, content_seg)
                    LPIPS.append(dist_model.forward(outputs, output2).squeeze())
                # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
                basename = os.path.basename(names[1])
                path = os.path.join(opts.output_folder+"_%02d"%j,basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if opts.compute_CIS:
                cur_preds = np.concatenate(cur_preds, 0)
                py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
                for j in range(cur_preds.shape[0]):
                    pyx = cur_preds[j, :]
                    CIS.append(entropy(pyx, py))
            if not opts.output_only:
                # also save input images
                vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
        if opts.compute_IS:
            all_preds = np.concatenate(all_preds, 0)
            py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
            for j in range(all_preds.shape[0]):
                pyx = all_preds[j, :]
                IS.append(entropy(pyx, py))

        if opts.compute_IS:
            print("Inception Score: {}".format(np.exp(np.mean(IS))))
        if opts.compute_CIS:
            print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
        if opts.compute_LPIPS:
            print("Average LPIPS distance: {}".format(torch.mean(torch.stack(LPIPS))))

elif opts.trainer == 'MUNIT':
    with torch.no_grad():
        # Start testing
        style_fixed = Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
        for i, (images, names) in enumerate(zip(data_loader, image_names)):
            if opts.compute_LPIPS and i > 200:
                break
            if opts.compute_CIS:
                cur_preds = []

            print(names[1])
            images = Variable(images.cuda(), volatile=True)
            content, _ = encode(images)
            style = style_fixed if opts.synchronized else Variable(torch.randn(opts.num_style, style_dim, 1, 1).cuda(), volatile=True)
            for j in range(opts.num_style):
                s = style[j].unsqueeze(0)
                outputs = decode(content, s)
                outputs = (outputs + 1) / 2.
                if opts.compute_IS or opts.compute_CIS:
                    pred = F.softmax(inception(inception_up(outputs)), dim=1).cpu().data.numpy()  # get the predicted class distribution
                if opts.compute_IS:
                    all_preds.append(pred)
                if opts.compute_CIS:
                    cur_preds.append(pred)
                if opts.compute_LPIPS:
                    rand_code = torch.randn(1, style_dim, 1, 1).cuda()
                    output2 = decode(content, rand_code)
                    LPIPS.append(dist_model.forward(outputs, output2).squeeze())
                # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
                basename = os.path.basename(names[1])
                path = os.path.join(opts.output_folder+"_%02d"%j,basename)
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                vutils.save_image(outputs.data, path, padding=0, normalize=True)
            if opts.compute_CIS:
                cur_preds = np.concatenate(cur_preds, 0)
                py = np.sum(cur_preds, axis=0)  # prior is computed from outputs given a specific input
                for j in range(cur_preds.shape[0]):
                    pyx = cur_preds[j, :]
                    CIS.append(entropy(pyx, py))
            if not opts.output_only:
                # also save input images
                vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
        if opts.compute_IS:
            all_preds = np.concatenate(all_preds, 0)
            py = np.sum(all_preds, axis=0)  # prior is computed from all outputs
            for j in range(all_preds.shape[0]):
                pyx = all_preds[j, :]
                IS.append(entropy(pyx, py))

        if opts.compute_IS:
            print("Inception Score: {}".format(np.exp(np.mean(IS))))
        if opts.compute_CIS:
            print("conditional Inception Score: {}".format(np.exp(np.mean(CIS))))
        if opts.compute_LPIPS:
            print("Average LPIPS distance: {}".format(torch.mean(torch.stack(LPIPS))))

elif opts.trainer == 'UNIT':
    # Start testing
    for i, (images, names) in enumerate(zip(data_loader, image_names)):
        print(names[1])
        images = Variable(images.cuda(), volatile=True)
        content, _ = encode(images)

        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        # path = os.path.join(opts.output_folder, 'input{:03d}_output{:03d}.jpg'.format(i, j))
        basename = os.path.basename(names[1])
        path = os.path.join(opts.output_folder,basename)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        vutils.save_image(outputs.data, path, padding=0, normalize=True)
        if not opts.output_only:
            # also save input images
            vutils.save_image(images.data, os.path.join(opts.output_folder, 'input{:03d}.jpg'.format(i)), padding=0, normalize=True)
else:
    pass

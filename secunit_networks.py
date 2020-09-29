"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from models.architecture import SPADEResnetBlock
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

import random

##################################################################################
# Discriminator
##################################################################################

class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, input_dim, params):
        super(MsImageDis, self).__init__()
        self.n_layer = params['n_layer']
        self.gan_type = params['gan_type']
        self.dim = params['dim']
        self.norm = params['norm']
        self.activ = params['activ']
        self.num_scales = params['num_scales']
        self.pad_type = params['pad_type']
        self.input_dim = input_dim
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.upsample = nn.Upsample(scale_factor=2.0)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())
        self.ups_cnn = self._make_net()

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 4, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 4, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        h, w = x.size()[2], x.size()[3]
        h_idx,w_idx = random.randint(0,h), random.randint(0,w)

        ups_x = self.upsample(x)[:,:,h_idx:h+h_idx,w_idx:w+w_idx]
        outputs.append(self.ups_cnn(ups_x))

        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def make_one_hot(self, labels, C=2):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
    
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        target = torch.autograd.Variable(target)
        return target

    def calc_dis_loss(self, input_fake, input_real, real_to_one_hot=False):
        # calculate the loss to train D
        if real_to_one_hot:
            input_real = self.make_one_hot(input_real, C=input_fake.size()[1])

        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

##################################################################################
# Generator
##################################################################################

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, seg_dim, params):
        super(AdaINGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        self.n_seg_classes = seg_dim
        spade_config = params['spade_config']

        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim // 2, 'in', activ, pad_type=pad_type)
        self.dec = StyleBasedDecoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, seg_dim, spade_config=spade_config, res_norm='adain', activ=activ, pad_type=pad_type) # res_norm = 'adain'
        
        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)
    
        self.enc_seg = ContentEncoder(n_downsample, n_res, seg_dim, dim // 2, 'in', activ, pad_type=pad_type) # input_dim -> seg_dim
        self.dec_seg = Decoder(n_downsample, n_res, self.enc_content.output_dim, seg_dim, res_norm='in', activ=activ, pad_type=pad_type, seg_mode=True)

        #self.final_dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        
        self.share_translators = params['share_translators']
        self.shared_content = params['shared_content']
        self.translate_size = int(dim * 2 * (1 - self.shared_content))
        if self.shared_content < 1.0:
            self.translate_c = MLP(self.translate_size, self.translate_size, self.translate_size, 2, norm='in', activ=activ)
            if not self.share_translators:    
                self.translate_seg = MLP(self.translate_size, self.translate_size, self.translate_size, 2, norm='in', activ=activ)
            else:
                self.translate_seg = self.translate_c
            
        #self.translate_c_a = MLP(translate_size, translate_size, translate_size, 2, norm='in', activ=activ) 
        #self.translate_seg_a = MLP(translate_size, translate_size, translate_size, 2, norm='in', activ=activ)

    def translate(self, code, seg_code):
        if self.shared_content == 1.0:
            return code, seg_code

        if self.shared_content == 0.0:
            return self.translate_c(code), self.translate_seg(seg_code)

        code = torch.cat([self.translate_c(code[:, :self.translate_size]), code[:, self.translate_size:]], dim=1)
        seg_code = torch.cat([self.translate_seg(seg_code[:, :self.translate_size]), seg_code[:, self.translate_size:]], dim=1)
        return code, seg_code

    def forward(self, images, seg_imgs):
        # reconstruct an image
        content, style_fake, seg_content = self.encode(images, seg_imgs)
        images_recon, _, _ = self.decode(content, style_fake, seg_content, seg_imgs)
        return images_recon

    def encode(self, images, seg_imgs):
        # encode an image to its content and style codes
        channels = seg_imgs.size(1)
        if channels == 1:
            seg_one_hot = self.make_one_hot(seg_imgs)
        else:
            assert (channels == self.n_seg_classes)
            seg_one_hot = seg_imgs

        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        seg_content = self.enc_seg(seg_one_hot)
        return content, style_fake, seg_content

    def decode(self, content, style, seg_content, seg_imgs):
        channels = seg_imgs.size(1)
        if channels == 1:
            seg_one_hot = self.make_one_hot(seg_imgs)
        else:
            assert (channels == self.n_seg_classes)
            seg_one_hot = seg_imgs

        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        feat_final, _, _ = self.dec(seg_content, content, seg_one_hot)
        return feat_final
    
    def make_one_hot(self, labels):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
    
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        one_hot = torch.cuda.FloatTensor(labels.size(0), self.n_seg_classes, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        target = torch.autograd.Variable(target)
        return target

    def old_decode(self, content, style, seg_content):
        # decode content and style codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.final_dec)
        self.assign_adain_params(adain_params, self.dec)
        feat_content = content
        feat_final = seg_content
        for i in range(len(self.dec.model)):
            if isinstance(self.dec.model[i], ResBlocks):
                for j in range(len(self.dec.model[i].model)):
                    feat_content = self.dec.model[i].model[j].forward(feat_content)
                    if j > -1: # best was 1 so far
                        feat_final = feat_content + self.final_dec.model[i].model[j].forward(feat_final)
                    else:
                        feat_final = self.final_dec.model[i].model[j].forward(feat_final)
            elif isinstance(self.dec.model[i], Conv2dBlock):
                feat_content = self.dec.model[i].forward(feat_content)
                feat_final = self.final_dec.model[i].forward(feat_final)
            else:
                feat_content = self.dec.model[i].forward(feat_content)
                feat_final = self.final_dec.model[i].forward(feat_final) 
        return feat_final

    def decode_seg(self, content):
        images = self.dec_seg(content)
        return images

    def encode_seg(self, images):
        enc = self.enc_seg(images)
        return enc

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params


class VAEGen(nn.Module):
    # VAE architecture
    def __init__(self, input_dim, params):
        super(VAEGen, self).__init__()
        dim = params['dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']

        # content encoder
        self.enc = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
        self.dec = Decoder(n_downsample, n_res, self.enc.output_dim, input_dim, res_norm='in', activ=activ, pad_type=pad_type)

    def forward(self, images):
        # This is a reduced VAE implementation where we assume the outputs are multivariate Gaussian distribution with mean = hiddens and std_dev = all ones.
        hiddens = self.encode(images)
        if self.training == True:
            noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
            images_recon = self.decode(hiddens + noise)
        else:
            images_recon = self.decode(hiddens)
        return images_recon, hiddens

    def encode(self, images):
        hiddens = self.enc(images)
        noise = Variable(torch.randn(hiddens.size()).cuda(hiddens.data.get_device()))
        return hiddens, noise

    def decode(self, hiddens):
        images = self.dec(hiddens)
        return images


##################################################################################
# Encoder and Decoders
##################################################################################

class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)

class StyleBasedDecoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, n_labels, spade_config='spectralspadeadain3x3',res_norm='adain', activ='relu', pad_type='zero'):
        super(StyleBasedDecoder, self).__init__()

        self.model = []
        # SPADE residual blocks
        self.model += [StyleBasedBlocks(n_res, dim, n_labels, res_norm, activ, pad_type, spade_config)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [UpsampleBlock(),
                           Conv2dWrapper(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dWrapper(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]

        self.model = MultInputSequential(*self.model)

    def forward(self, final_feats, intermed, seg_imgs):
        return self.model(final_feats, intermed, seg_imgs)

class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, res_norm='adain', activ='relu', pad_type='zero', seg_mode=False):
        super(Decoder, self).__init__()

        self.model = []
        # AdaIN residual blocks
        self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # upsampling blocks
        for i in range(n_upsample):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
            dim //= 2
        # use reflection padding in the last conv layer
        if not seg_mode:
            self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        else:
            self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='sigm', pad_type=pad_type)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


##################################################################################
# Sequential Models
##################################################################################
class StyleBasedBlocks(nn.Module):
    def __init__(self, num_blocks, dim, n_labels, norm, activation, pad_type, spade_config):
        super(StyleBasedBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [StyleBasedBlock(dim, n_labels, norm, activation, pad_type, spade_config)]
        self.model = MultInputSequential(*self.model)

    def forward(self, final_feats, intermed, seg_imgs):
        return self.model(final_feats, intermed, seg_imgs)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        b, c, h, w = x.size()
        #hacky, but allows MLP class to be used with 3d tensors in the way I need
        if h == 1 and w == 1:
            return self.model(x.view(x.size(0), -1))
        else:
            return self.model(x)

##################################################################################
# Basic Blocks
##################################################################################
class MultInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input

class UpsampleBlock(nn.Module):
    def __init__(self, scale_factor=2.0):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
    
    def forward(self, final_feats, intermed, seg_imgs):
        return self.upsample(final_feats), self.upsample(intermed), seg_imgs

class StyleBasedBlock(nn.Module):
    def __init__(self, dim, n_labels, norm, activation, pad_type, spade_config):
        super(StyleBasedBlock, self).__init__()

        self.final_model = SPADEResnetBlock(dim, dim, spade_config, n_labels)
        self.intermed_model = ResBlock(dim, norm, activation, pad_type)
        #self.intermed_model = SPADEResBlock(dim, dim, "adainspadesyncbatch3x3", n_labels)

    def forward(self, final_feats, intermed, seg_imgs):
        intermed = self.intermed_model(intermed)
        return intermed + self.final_model(final_feats, seg_imgs), intermed, seg_imgs 

class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out

class Conv2dWrapper(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dWrapper, self).__init__()
        self.conv_block = Conv2dBlock(input_dim, output_dim, kernel_size, stride, padding, norm, activation, pad_type)
    
    def forward(self, final_feats, intermed, seg_imgs):
        return self.conv_block(final_feats), self.conv_block(intermed), seg_imgs


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim) #, track_running_stats=True)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        elif activation == 'sigm':
            self.activation = nn.Sigmoid()
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        condense = False
        if len(x.size()) == 4:
            condense = True
            b_size, c, h, w = x.size()
            x = x.view(b_size, -1, c)
        out = self.fc(x)

        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        
        if condense:
            out = out.view(b_size, c, h, w)

        return out

##################################################################################
# VGG network definition
##################################################################################
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        # relu1_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        # relu2_2 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        # relu3_3 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        # relu4_3 = h

        h = F.relu(self.conv5_1(h), inplace=True)
        h = F.relu(self.conv5_2(h), inplace=True)
        h = F.relu(self.conv5_3(h), inplace=True)
        relu5_3 = h

        return relu5_3
        # return [relu1_2, relu2_2, relu3_3, relu4_3]

##################################################################################
# Normalization layers
##################################################################################
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

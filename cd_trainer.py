
from cd_networks import AdaINGen, MsImageDis, VAEGen
from cd_utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
import torch
import torch.nn as nn
import os

class CDUNIT_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(CDUNIT_Trainer, self).__init__()
        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = AdaINGen(hyperparameters['input_dim_a'], hyperparameters['n_seg_classes_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.gen_b = AdaINGen(hyperparameters['input_dim_b'], hyperparameters['n_seg_classes_b'], hyperparameters['gen'])  # auto-encoder for domain b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.dis_seg_a = MsImageDis(hyperparameters['n_seg_classes_a'], hyperparameters['dis'])
        self.dis_seg_b = MsImageDis(hyperparameters['n_seg_classes_b'], hyperparameters['dis'])

        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        self.style_dim = hyperparameters['gen']['style_dim']

        # fix the noise used in sampling
        self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
        self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters()) + list(self.dis_seg_a.parameters()) + list(self.dis_seg_b.parameters())
        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))
        self.dis_seg_a.apply(weights_init('gaussian'))
        self.dis_seg_b.apply(weights_init('gaussian'))

        self.cross_entropy = nn.CrossEntropyLoss(reduction='elementwise_mean')

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        s_a = Variable(self.s_a)
        s_b = Variable(self.s_b)
        c_a, s_a_fake, c_a_seg = self.gen_a.encode(x_a)
        c_b, s_b_fake, c_b_seg = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(c_b, s_a, c_b_seg)
        x_ab = self.gen_b.decode(c_a, s_b, c_a_seg)
        self.train()
        return x_ab, x_ba

    def gen_update(self, x_a, real_a_seg, x_b, real_b_seg, hyperparameters):
        self.gen_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, s_a_prime, c_a_seg = self.gen_a.encode(x_a)
        c_b, s_b_prime, c_b_seg = self.gen_b.encode(x_b)
        # segmentations
        x_a_seg = self.gen_a.dec_seg(c_a_seg)
        x_b_seg = self.gen_b.dec_seg(c_b_seg)        
        
        # decode (within domain)
        x_a_recon = self.gen_a.decode(c_a, s_a_prime, c_a_seg)
        x_b_recon = self.gen_b.decode(c_b, s_b_prime, c_b_seg)
        # decode (cross domain)

        c_ab, c_ab_seg = self.gen_b.translate(c_a, c_a_seg) 
        c_ba, c_ba_seg = self.gen_a.translate(c_b, c_b_seg)

        x_ab_seg = self.gen_b.dec_seg(c_ab_seg)
        x_ba_seg = self.gen_a.dec_seg(c_ba_seg)

        x_ba = self.gen_a.decode(c_ba, s_a, c_ba_seg) #was c_b
        x_ab = self.gen_b.decode(c_ab, s_b, c_ab_seg) #was c_a

        # encode again
        c_ba_recon, s_a_recon, c_ba_seg_recon = self.gen_a.encode(x_ba) #was c_b_recon
        c_ab_recon, s_b_recon, c_ab_seg_recon = self.gen_b.encode(x_ab) #was c_a_recon

        c_a_recon, c_a_seg_recon = self.gen_a.translate(c_ab_recon, c_ab_seg_recon)
        c_b_recon, c_b_seg_recon = self.gen_b.translate(c_ba_recon, c_ba_seg_recon)

        # decode again (if needed)
        x_aba = self.gen_a.decode(c_a_recon, s_a_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(c_b_recon, s_b_prime) if hyperparameters['recon_x_cyc_w'] > 0 else None

        # reconstruction loss
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_s_a = self.recon_criterion(s_a_recon, s_a)
        self.loss_gen_recon_s_b = self.recon_criterion(s_b_recon, s_b)
        self.loss_gen_recon_c_a = self.recon_criterion(c_a_recon, c_a)
        self.loss_gen_recon_c_b = self.recon_criterion(c_b_recon, c_b)
        self.loss_gen_recon_intermed1 = self.recon_criterion(c_ba_seg_recon, c_ba_seg)
        self.loss_gen_recon_intermed2 = self.recon_criterion(c_ab_seg_recon, c_ab_seg)
        self.loss_gen_recon_intermed3 = self.recon_criterion(c_ba_recon, c_ba)
        self.loss_gen_recon_intermed4 = self.recon_criterion(c_ab_recon, c_ab)

        self.loss_gen_recon_seg_a = self.cross_entropy(x_a_seg, real_a_seg.view(real_a_seg.size()[0], real_a_seg.size()[2], real_a_seg.size()[3])) #self.recon_criterion(x_a_seg, real_a_seg)
        self.loss_gen_recon_seg_b = self.cross_entropy(x_b_seg, real_b_seg.view(real_b_seg.size()[0], real_b_seg.size()[2], real_b_seg.size()[3])) #self.recon_criterion(x_b_seg, real_b_seg)

        self.loss_gen_recon_cseg_a = self.recon_criterion(c_a_seg_recon, c_a_seg)
        self.loss_gen_recon_cseg_b = self.recon_criterion(c_b_seg_recon, c_b_seg)

        self.loss_gen_cycrecon_x_a = self.recon_criterion(x_aba, x_a) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        self.loss_gen_cycrecon_x_b = self.recon_criterion(x_bab, x_b) if hyperparameters['recon_x_cyc_w'] > 0 else 0
        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)
        self.loss_gen_adv_seg_a = self.dis_seg_a.calc_gen_loss(x_a_seg) + self.dis_seg_a.calc_gen_loss(x_ba_seg)
        self.loss_gen_adv_seg_b = self.dis_seg_b.calc_gen_loss(x_b_seg) + self.dis_seg_b.calc_gen_loss(x_ab_seg)

        # domain-invariant perceptual loss
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0
        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_a + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_a + \
                              hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                              hyperparameters['recon_s_w'] * self.loss_gen_recon_s_b + \
                              hyperparameters['recon_c_w'] * self.loss_gen_recon_c_b + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_a + \
                              hyperparameters['recon_x_cyc_w'] * self.loss_gen_cycrecon_x_b + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                              hyperparameters['vgg_w'] * self.loss_gen_vgg_b 
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_seg_a + \
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_seg_b + \
                              #hyperparameters['recon_cseg_w'] * self.loss_gen_recon_cseg_a + \
                              #hyperparameters['recon_cseg_w'] * self.loss_gen_recon_cseg_b + \
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed1 + \
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed2 + \
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed3 + \
                              #hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed4 
                              #hyperparameters['gan_w_seg'] * self.loss_gen_adv_seg_a + hyperparameters['gan_w_seg'] * self.loss_gen_adv_seg_b



        self.loss_gen_total.backward(retain_graph=True)
        self.gen_opt.step()

        
        self.loss_gen_total_overflow = hyperparameters['recon_seg_w'] * self.loss_gen_recon_seg_a + hyperparameters['recon_seg_w'] * self.loss_gen_recon_seg_b
        self.loss_gen_total_overflow += hyperparameters['recon_cseg_w'] * self.loss_gen_recon_cseg_a + hyperparameters['recon_cseg_w'] * self.loss_gen_recon_cseg_b
        self.loss_gen_total_overflow += hyperparameters['gan_w_seg'] * self.loss_gen_adv_seg_a + hyperparameters['gan_w_seg'] * self.loss_gen_adv_seg_b
        self.loss_gen_total_overflow += hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed1 + \
                              hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed2 + \
                              hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed3 + \
                              hyperparameters['recon_seg_w'] * self.loss_gen_recon_intermed4 
        self.loss_gen_total_overflow.backward()
        self.gen_opt.step()


    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg.features(img_vgg)
        target_fea = vgg.features(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def segment_a(self, x_a):
        x_a_gen = []
        for i in range(x_a.size(0)):
            c_a, s_a_fake, c_a_seg = self.gen_a.encode(x_a[i].unsqueeze(0))
            x_a_gen.append(self.gen_a.decode_seg(c_a_seg))
        return x_a_gen

    def segment_b(self, x_b):
        x_b_gen = []
        for i in range(x_b.size(0)):
            c_b, s_b_fake, c_b_seg = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_b_gen.append(self.gen_b.decode_seg(c_b_seg))
        return x_b_gen

    def segment_ab(self, x_a, x_b):
        x_a_gen = []
        x_b_gen = []
        for i in range(x_a.size(0)):
            c_a, s_a_fake, c_a_seg = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake, c_b_seg = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_gen.append(self.gen_a.decode_seg(c_a_seg))
            x_b_gen.append(self.gen_b.decode_seg(c_b_seg))
        return x_a_gen, x_b_gen
    
    def sample(self, x_a, x_b, n_random_codes=10):
        x_ab_rand, x_ba_rand = [],[]
        for i in range(n_random_codes):
            x_ab_rand.append([])
            x_ba_rand.append([])
        s_a_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())
        s_b_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())

        self.eval()
        s_a1 = Variable(self.s_a)
        s_b1 = Variable(self.s_b)
        s_a2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_b_recon, x_ba1, x_ba2, x_ab1, x_ab2, x_a_seg, x_b_seg = [], [], [], [], [], [], [], []
        if len(x_a.size()) == 3:
            x_a = x_a.unsqueeze(0)
        if len(x_b.size()) == 3:
            x_b = x_b.unsqueeze(0)
        for i in range(x_a.size(0)):
            c_a, s_a_fake, c_a_seg = self.gen_a.encode(x_a[i].unsqueeze(0))
            c_b, s_b_fake, c_b_seg = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake, c_a_seg))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake, c_b_seg))
            #x_a_seg.append(self.gen_a.decode_seg(c_a_seg))
            #x_b_seg.append(self.gen_b.decode_seg(c_b_seg))
            
            c_ab, c_ab_seg = self.gen_b.translate(c_a, c_a_seg)
            c_ba, c_ba_seg = self.gen_a.translate(c_b, c_b_seg)
            x_a_seg.append(self.gen_b.dec_seg(c_ab_seg))
            x_b_seg.append(self.gen_a.dec_seg(c_ba_seg))

            x_ba1.append(self.gen_a.decode(c_ba, s_a1[i].unsqueeze(0), c_ba_seg))
            x_ba2.append(self.gen_a.decode(c_ba, s_a2[i].unsqueeze(0), c_ba_seg))
            x_ab1.append(self.gen_b.decode(c_ab, s_b1[i].unsqueeze(0), c_ab_seg))
            x_ab2.append(self.gen_b.decode(c_ab, s_b2[i].unsqueeze(0), c_ab_seg))
            for n in range(n_random_codes):
                x_ab_rand[n].append(self.gen_b.decode(c_ab, s_b_rand[n].unsqueeze(0), c_ab_seg))
                x_ba_rand[n].append(self.gen_a.decode(c_ba, s_a_rand[n].unsqueeze(0), c_ba_seg))

        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        for i in range(n_random_codes):
            x_ab_rand[i] = torch.cat(x_ab_rand[i])
            x_ba_rand[i] = torch.cat(x_ba_rand[i])
        self.train()
        return (x_a, x_a_recon, x_ab1, x_ab2, *x_ab_rand, x_b, x_b_recon, x_ba1, x_ba2, *x_ba_rand), (torch.cat(x_a_seg), torch.cat(x_b_seg))


    def dis_update(self, x_a, x_a_seg, x_b, x_b_seg, hyperparameters):
        self.dis_opt.zero_grad()
        s_a = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        s_b = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        # encode
        c_a, _, c_a_seg = self.gen_a.encode(x_a)
        c_b, _, c_b_seg = self.gen_b.encode(x_b)
        # segmentation
        fake_a_seg = self.gen_a.dec_seg(c_a_seg)
        fake_b_seg = self.gen_b.dec_seg(c_b_seg)

        c_ab, c_ab_seg = self.gen_b.translate(c_a, c_a_seg) 
        c_ba, c_ba_seg = self.gen_a.translate(c_b, c_b_seg) 

        fake_ab_seg = self.gen_b.dec_seg(c_ab_seg)
        fake_ba_seg = self.gen_a.dec_seg(c_ba_seg)

        # decode (cross domain)
        x_ba = self.gen_a.decode(c_ba, s_a, c_ba_seg)
        x_ab = self.gen_b.decode(c_ab, s_b, c_ab_seg)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        #self.loss_dis_seg = self.dis_seg_a.calc_dis_loss(fake_a_seg.detach(), x_a_seg, True) 
        #self.loss_dis_seg += self.dis_seg_b.calc_dis_loss(fake_b_seg.detach(), x_b_seg, True) 
        self.loss_dis_seg = self.dis_seg_a.calc_dis_loss(fake_ba_seg.detach(), x_a_seg, True) 
        self.loss_dis_seg += self.dis_seg_b.calc_dis_loss(fake_ab_seg.detach(), x_b_seg, True)

        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b + hyperparameters['gan_w_seg'] * self.loss_dis_seg
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis_0")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])

        last_model_name = get_model_list(checkpoint_dir, "dis_seg")
        state_dict = torch.load(last_model_name)
        self.dis_seg_a.load_state_dict(state_dict['a'])
        self.dis_seg_b.load_state_dict(state_dict['b'])
        # Load optimizers
        try:
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.dis_opt.load_state_dict(state_dict['dis'])
            self.gen_opt.load_state_dict(state_dict['gen'])
        except:
            print("Unable to load optimizer")
        # Reinitialize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        dis_seg_name = os.path.join(snapshot_dir, 'dis_seg%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
        torch.save({'a': self.dis_seg_a.state_dict(), 'b': self.dis_seg_b.state_dict()}, dis_seg_name)

    def sample_a(self, x_a, n_random_codes=10):
        x_ab_rand, x_s_rand = [], []
        for i in range(n_random_codes):
            x_ab_rand.append([])
            x_s_rand.append([])
        s_a_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())
        s_b_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())

        self.eval()
        s_b1 = Variable(self.s_b)
        s_b2 = Variable(torch.randn(x_a.size(0), self.style_dim, 1, 1).cuda())
        x_a_recon, x_ab1, x_ab2, x_a_seg = [], [], [], []
        if len(x_a.size()) == 3:
            x_a = x_a.unsqueeze(0)

        for i in range(x_a.size(0)):
            c_a, s_a_fake, c_a_seg = self.gen_a.encode(x_a[i].unsqueeze(0))
            x_a_recon.append(self.gen_a.decode(c_a, s_a_fake, c_a_seg))
            #x_a_seg.append(self.gen_a.decode_seg(c_a_seg))

            c_ab, c_ab_seg = self.gen_b.translate(c_a, c_a_seg)
            x_a_seg.append(self.gen_b.dec_seg(c_ab_seg))

            x_ab1.append(self.gen_b.decode(c_ab, s_b1[i].unsqueeze(0), c_ab_seg))
            x_ab2.append(self.gen_b.decode(c_ab, s_b2[i].unsqueeze(0), c_ab_seg))
            for n in range(n_random_codes):
                x_ab_rand[n].append(self.gen_b.decode(c_ab, s_b_rand[n].unsqueeze(0), c_ab_seg))
                x_s_rand[n].append(self.gen_a.decode(c_a, s_a_rand[n].unsqueeze(0), c_a_seg)) 

        x_a_recon = torch.cat(x_a_recon)
        x_ab1, x_ab2 = torch.cat(x_ab1), torch.cat(x_ab2)
        for i in range(n_random_codes):
            x_ab_rand[i] = torch.cat(x_ab_rand[i])
            x_s_rand[i] = torch.cat(x_s_rand[i])
        self.train()
        return (x_a, x_a_recon, x_ab1, x_ab2, *x_ab_rand), (torch.cat(x_a_seg)), x_s_rand

    def sample_b(self, x_b, n_random_codes=10):
        x_ba_rand, x_s_rand = [], []
        for i in range(n_random_codes):
            x_ba_rand.append([])
            x_s_rand.append([])
        s_a_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())
        s_b_rand = Variable(torch.randn(n_random_codes, self.style_dim, 1, 1).cuda())

        self.eval()
        s_a1 = Variable(self.s_a)
        s_a2 = Variable(torch.randn(x_b.size(0), self.style_dim, 1, 1).cuda())
        x_b_recon, x_ba1, x_ba2, x_b_seg = [], [], [], []
        if len(x_b.size()) == 3:
            x_b = x_b.unsqueeze(0)

        for i in range(x_b.size(0)):
            c_b, s_b_fake, c_b_seg = self.gen_b.encode(x_b[i].unsqueeze(0))
            x_b_recon.append(self.gen_b.decode(c_b, s_b_fake, c_b_seg))
            #x_b_seg.append(self.gen_b.decode_seg(c_b_seg))

            c_ba, c_ba_seg = self.gen_a.translate(c_b, c_b_seg)
            x_b_seg.append(self.gen_a.dec_seg(c_ba_seg))

            x_ba1.append(self.gen_a.decode(c_ba, s_a1[i].unsqueeze(0), c_ba_seg))
            x_ba2.append(self.gen_a.decode(c_ba, s_a2[i].unsqueeze(0), c_ba_seg))
            for n in range(n_random_codes):
                x_ba_rand[n].append(self.gen_a.decode(c_ba, s_a_rand[n].unsqueeze(0), c_ba_seg))
                x_s_rand[n].append(self.gen_b.decode(c_b, s_b_rand[n].unsqueeze(0), c_b_seg))

        x_b_recon = torch.cat(x_b_recon)
        x_ba1, x_ba2 = torch.cat(x_ba1), torch.cat(x_ba2)
        for i in range(n_random_codes):
            x_ba_rand[i] = torch.cat(x_ba_rand[i])
            x_s_rand[i] = torch.cat(x_s_rand[i])
        self.train()
        return (x_b, x_b_recon, x_ba1, x_ba2, *x_ba_rand), (torch.cat(x_b_seg)), x_s_rand

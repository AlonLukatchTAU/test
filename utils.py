from imresize import imresize
from models import ContentLoss, ConvBlock, Generator
import math
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import copy



def adjust_scales2image(size, opt):
    opt.num_scales = math.ceil((math.log(math.pow(opt.min_size / size, 1), opt.scale_factor_init))) + 1
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    opt.scale1 = min(opt.max_size / size, 1)
    opt.scale_factor = math.pow(opt.min_size / size, 1 / opt.stop_scale)
    scale2stop = math.ceil(math.log(min([opt.max_size, size]) / size, opt.scale_factor_init))
    opt.stop_scale = opt.num_scales - scale2stop
    return

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    if type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)+5
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    if type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    return noise

def generate_noise2(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    noise = []
    for i in range(size[0]):
        noise.append(generate_noise(size[1:], num_samp=1, device='cuda', type='gaussian', scale=1).squeeze(0))

    res = torch.stack(noise, dim=0)

    return res


def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def load_trained_pyramid(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if(os.path.exists(opt.load)):
        Gs = torch.load('%s/Gs.pth' % opt.load, map_location=opt.device)
        Zs = torch.load('%s/Zs.pth' % opt.load)
        reals = torch.load('%s/reals.pth' % opt.load)
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % opt.load)
    else:
        print('no appropriate trained model is exist, please train first')
    opt.mode = mode
    return Gs,Zs,reals,NoiseAmp

def load_trained_pyramid_mix(opt, mode_='train'):
    mode = opt.mode
    opt.mode = 'train'
    if(os.path.exists(opt.load)):
        Gs_a = torch.load('%s/Gs_a.pth' % opt.load, map_location=opt.device)
        Zs_a = torch.load('%s/Zs_a.pth' % opt.load)
        reals_a = torch.load('%s/reals_a.pth' % opt.load)
        NoiseAmp_a = torch.load('%s/NoiseAmp_a.pth' % opt.load, map_location=opt.device)

        Gs_b = torch.load('%s/Gs_b.pth' % opt.load, map_location=opt.device)
        Zs_b = torch.load('%s/Zs_b.pth' % opt.load)
        reals_b = torch.load('%s/reals_b.pth' % opt.load)
        NoiseAmp_b = torch.load('%s/NoiseAmp_b.pth' % opt.load, map_location=opt.device)

    else:
        print('no appropriate trained model is exist, please train first')
        sys.exit()
    opt.mode = mode
    return Gs_a, Zs_a, reals_a, NoiseAmp_a, Gs_b, Zs_b, reals_b, NoiseAmp_b

def get_content_model_and_loss(opt, cnn, prev):
    cnn = copy.deepcopy(cnn)
    
    content_layer = 'conv_{}'.format(opt.content+1)
    
    model = nn.Sequential()
    
    i = 0

    for layer in cnn.modules():
        flag = True
        if isinstance(layer, ConvBlock):
            i += 1
            name = 'ConvBlock_{}'.format(i)
            flag = False
        elif isinstance(layer, nn.Conv2d):
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.LeakyReLU):
            name = 'LeakyReLu_{}'.format(i)
            layer = nn.LeakyReLU(inplace=False)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'norm_{}'.format(i)
        elif isinstance(layer, nn.Sequential):
            name = 'seq_{}'.format(i)
            flag = False
        elif isinstance(layer, nn.Tanh):
            name = 'tanh_{}'.format(i)
        elif isinstance(layer, Generator):
            name = 'generator'
            flag = False
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            
        
        if not(flag):
            continue
        
        model.add_module(name, layer)
        
        if name == content_layer:
            target = model(prev).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            break

           
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break
    model = model[:(i+1)]
    
    return model, content_loss

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def apply_content_loss(opt, cnn, img, num_steps=300):
    model, content_loss = get_content_model_and_loss(opt, cnn, img)
    optimizer = get_input_optimizer(img)
    run = [0]
    while run[0] <= num_steps:
        
        def closure():
            img.data.clamp_(0,1)
            
            optimizer.zero_grad()
            model(img)
            loss = content_loss.loss
            loss.backward()
            
            run[0] += 1
            return loss
        
        optimizer.step(closure)
        
    img.data.clamp_(0,1)
    
    return img
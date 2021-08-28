import time
import torch.backends.cudnn as cudnn
from torch import nn
import datetime
import matplotlib.pyplot as plt
import sys
import matplotlib.image as mpimg
import cv2 as cv
import os
import numpy as np
from dataloader import DataLoader
from utils import AverageMeter
from __future__ import print_function, division
import scipy
import scipy.misc
from models import Generator, Discriminator, ResidualBlock, VGG_19

# vars
kernel_one_g = 9
kernel_two_g = 3
scaling = 4
n_blocks_g = 16 
n_channels_g = 64 
lr = 1e-4
kernel_size_d = 3
n_channels_d = 64
n_blocks_d = 8
fl_size_d = 1024 
vgg_i = 5
vgg_j = 4
batch_size = 1
iterations = 2e5  
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
beta = 1e-3 
print_freq = 500 

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(loader, generator, discriminator, vgg19, loss_content, loss_adversarial, 
          optimizer_g, optimizer_d, epoch, batch_size):

  generator.train()
  discriminator.train()

  batch_time = AverageMeter()  # forward prop. + back prop. time
  data_time = AverageMeter() 
  losses_c = AverageMeter()  # content loss
  losses_a = AverageMeter()  # adversarial loss in the generator
  losses_d = AverageMeter() 

  start = time.time()
  imgs_hr, imgs_lr = loader.load_data(batch_size = batch_size)
  for i in range(10):
    imgs_lr = torch.tensor(imgs_lr).to(device)
    imgs_hr = torch.tensor(imgs_hr).to(device)


    imgs_lr = imgs_lr.view(1,3,64,64)
    imgs_hr = imgs_hr.view(1,3,256,256)
    
    #generate sr images
    gen_sr = generator(imgs_lr.float())

    #convert images to 3 layered
    gen_sr = (gen_sr + 1.) / 2.
    if gen_sr.ndimension() == 3:
        gen_sr = (gen_sr - imagenet_mean) / imagenet_std
    elif gen_sr.ndimension() == 4:
        gen_sr = (gen_sr - imagenet_mean_cuda) / imagenet_std_cuda

    gen_img_vgg = vgg19(gen_sr)
    hr_img_vgg = vgg19(imgs_hr.float())

    discriminator_sr = discriminator(gen_sr)

    content_loss = loss_content(gen_img_vgg, hr_img_vgg)
    adversarial_loss = loss_adversarial(discriminator_sr, torch.ones_like(discriminator_sr))
    perpectual_loss = content_loss + beta*adversarial_loss

    optimizer_g.zero_grad()
    perpectual_loss.backward()
    optimizer_g.step()
    
    losses_c.update(content_loss.item(), imgs_lr.size(0))
    losses_a.update(adversarial_loss.item(), imgs_lr.size(0))

    hr_dis = discriminator(imgs_hr.float())
    discriminator_sr = discriminator(gen_sr.detach())

    adversarial_loss = loss_adversarial(discriminator_sr, torch.ones_like(discriminator_sr)) + \
                        loss_adversarial(hr_dis, torch.ones_like(hr_dis))

    optimizer_d.zero_grad()
    adversarial_loss.backward()

    optimizer_d.step()
    losses_d.update(adversarial_loss.item(), imgs_hr.size(0))

    batch_time.update(time.time() - start)

    # Reset start time
    start = time.time()

    # Print status
    if i % print_freq == 0:
      print('Epoch: [{0}][{1}]----'
            'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
            'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
            'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})----'
            'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----'
            'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                    len(imgs_lr),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time,
                                                                    loss_c=losses_c,
                                                                    loss_a=losses_a,
                                                                    loss_d=losses_d))

def buildModel():
  generator = Generator(kernel_one = kernel_one_g, kernel_two = kernel_two_g,
                        n_channels = n_channels_g, n_blocks = n_blocks_g, scaling_factor = scaling)
  
  optimizer_g = torch.optim.Adam(params = filter(lambda p:p.requires_grad, generator.parameters()), lr = lr)

  discriminator = Discriminator(kernel_size = kernel_size_d, n_channels = n_channels_d, 
                                n_blocks = n_blocks_d, fl_size = fl_size_d)
  
  optimizer_d = torch.optim.Adam(params=filter(lambda p:p.requires_grad, discriminator.parameters()),lr=lr)

  vgg_19 = VGG_19(vgg_i, vgg_j)
  vgg_19.eval()

  #Loss = content + adversarial
  loss_content = nn.MSELoss()
  loss_adversarial = nn.BCEWithLogitsLoss()

  #Move to default device
  generator = generator.to(device)
  discriminator = discriminator.to(device)
  vgg_19 = vgg_19.to(device)
  loss_content = loss_content.to(device)
  loss_adversarial = loss_adversarial.to(device)

  epochs = 1000

  data_loader = DataLoader(dataset_name = 'img_align_celeba', 
                                  img_res = (256, 256))
  
  #imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

  for epoch in range(0, epochs):
    train(loader = data_loader, generator = generator,
          discriminator = discriminator,
          vgg19 = vgg_19,
          loss_content = loss_content,
          loss_adversarial = loss_adversarial,
          optimizer_g = optimizer_g,
          optimizer_d = optimizer_d,
          epoch = epoch, 
          batch_size = batch_size)
    
    torch.save({'Epoch': epoch,
                'generator': generator,
                'discriminator': discriminator,
                'optimizer_g': optimizer_g,
                'optimizer_d': optimizer_d},
                'checkpoint_srgan.pth.tar')


                                                                    loss_d=losses_d))
if __name__ == '__main__':
  buildModel()
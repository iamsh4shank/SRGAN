import torch 
from torch import nn
import torchvision
import math

class ConvBlock(nn.Module):
  def __init__(self, input_ch, output_ch, kernel_size, stride = 1, bn = False, activation = None, pixel_shuffle = False, scaling_factor = None):
    super(ConvBlock, self).__init__()
    layers = []
    if (scaling_factor == None):
      layers.append(nn.Conv2d(input_ch, output_ch, kernel_size = kernel_size, stride = stride, padding = kernel_size // 2))
    else:
      layers.append(nn.Conv2d(input_ch, output_ch * (scaling_factor**2), kernel_size, padding = kernel_size // 2))
      layers.append(nn.PixelShuffle(upscale_factor = scaling_factor ))
    
    #adding bn layer if True 
    if bn is True:
      layers.append(nn.BatchNorm2d(num_features = output_ch, momentum = 0.8))
    
    #adding activation layer if activation is not None
    if activation == 'LeakyReLU':
      layers.append(nn.LeakyReLU(0.2))
    elif activation == 'PReLU':
      layers.append(nn.PReLU())
    elif activation == 'tanh':
      layers.append(nn.Tanh())
    #print (layers)
    self.conv_block = nn.Sequential(*layers)

  def forward(self, input):
    #print ('conv')
    #print(input.shape)
    output = self.conv_block(input)
    return output

class ResidualBlock(nn.Module):
  def __init__(self, kernel_size = 3, input_ch = 64, output_ch = 64):
    super(ResidualBlock, self).__init__()
    '''layers = []
    layers.append(nn.Conv2D(input_ch, output_ch, kernel_size = 3, stride = 1, padding = 'same')
    layers.append(nn.BatchNorm2D(num_features = output_ch, momentum = 0.8))
    layers.append(nn.PReLU())
    layers.append(nn.Conv2D(input_ch, output_ch, kernel_size = 3, stride = 1, padding = 'same')
    layers.append(nn.BatchNorm2D(num_features = output_ch, momentum = 0.8))'''
    self.conv_block1 = ConvBlock(input_ch = input_ch, output_ch = output_ch, kernel_size = kernel_size, bn = True, activation = 'PReLU', scaling_factor = None, pixel_shuffle= False)
    self.conv_block2 = ConvBlock(input_ch, output_ch, kernel_size = kernel_size, bn = True, activation = None)

  def forward(self, input):
    #print ('residual')
    #print (input.shape)
    residual = input
    #print ('outresit')
    #print (residual.shape)
    out = self.conv_block1(input)
    out = self.conv_block2(out)
    out = out + residual
  
    return out


class Generator(nn.Module):
  def __init__(self, kernel_one = 9, kernel_two = 3, n_channels = 64, n_blocks = 16, scaling_factor = 4):
    super(Generator, self).__init__()

    self.conv_block1 = ConvBlock(input_ch = 3, output_ch = n_channels, 
                                 kernel_size = kernel_one, bn = False, 
                                 activation = 'PReLU', 
                                 pixel_shuffle = False, scaling_factor = None )
    
    self.residual_block = nn.Sequential(*[ResidualBlock(kernel_size = kernel_two, 
                                                        input_ch = n_channels, 
                                                        output_ch = n_channels) for _ in range(n_blocks)])
    
    self.conv_block2 = ConvBlock(input_ch = n_channels, output_ch = n_channels, 
                                 kernel_size = kernel_two, bn = True, 
                                 activation = None)
    
    self.subpixel_block = nn.Sequential(*[ConvBlock(input_ch = n_channels, output_ch = n_channels, kernel_size = kernel_two, bn = False, activation = 'PReLU', scaling_factor = 2, pixel_shuffle = True)
                                      for _ in range(int(math.log2(scaling_factor)))])
    
    self.conv_block3 = ConvBlock(input_ch = n_channels, output_ch = 3, kernel_size = kernel_one, bn = False, activation = 'tanh')

  def forward(self, input):
    output = self.conv_block1(input)
    residual = output
    output = self.residual_block(output)
    output = self.conv_block2(output)
    output = output + residual
    output = self.subpixel_block(output)
    output = self.conv_block3(output)

    return output


class Discriminator(nn.Module):
  def __init__(self, kernel_size = 3, n_channels = 64, n_blocks = 8, fl_size = 1024):
    super(Discriminator, self).__init__()
    self.conv_block1 = ConvBlock(input_ch = 3, output_ch = 64, kernel_size = kernel_size, bn = False, activation = 'LeakyReLU')
    self.conv_block2 = ConvBlock(input_ch = 64, output_ch = 64, kernel_size = kernel_size, stride = 2, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block3 = ConvBlock(input_ch = 64, output_ch = 128, kernel_size = kernel_size, bn = True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block4 = ConvBlock(input_ch = 128, output_ch = 128, kernel_size = kernel_size, stride = 2, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block5 = ConvBlock(input_ch = 128, output_ch = 256, kernel_size = kernel_size, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block6 = ConvBlock(input_ch = 256, output_ch = 256, kernel_size = kernel_size, stride = 2, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block7 = ConvBlock(input_ch = 256, output_ch = 512, kernel_size = kernel_size, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)
    self.conv_block8 = ConvBlock(input_ch = 512, output_ch = 512, kernel_size = kernel_size, stride = 2, bn =  True, activation = 'LeakyReLU', pixel_shuffle = False)

    self.pool = nn.AdaptiveAvgPool2d((6,6))
    self.fc1 = nn.Linear(512*6*6, fl_size)
    self.leaky = nn.LeakyReLU(0.2)
    self.fc2 = nn.Linear(n_channels*16, 1)

  def forward(self, input):
    batch_size = input.size(0)  #1
    output = self.conv_block1(input)
    output = self.conv_block2(output)
    output = self.conv_block3(output)
    output = self.conv_block4(output)
    output = self.conv_block5(output)
    output = self.conv_block6(output)
    output = self.conv_block7(output)
    output = self.conv_block8(output)
    output = self.pool(output)
    output = self.fc1(output.view(batch_size, -1))
    output = self.leaky(output)
    output = self.fc2(output)

    return output


class VGG_19(nn.Module):
  def __init__(self, i = 5, j = 4):
    super(VGG_19, self).__init__()
    vgg = torchvision.models.vgg19(pretrained = True)
    #we need to iterated to particular feature layer

    conv_c,pool_c, truncate_c = 0,0,0
    for k in vgg.features.children():
      truncate_c += 1
      #check for layer type
      if isinstance(k, nn.Conv2d):
        conv_c += 1
      if isinstance(k, nn.MaxPool2d):
        pool_c += 1
      
      #break if reached to i,j according to paper
      if  pool_c == i-1 and conv_c == j:
        break

      #required vgg layer by the help of truncted counter
      self.vgg_19 = nn.Sequential(*list(vgg.features.children())[:truncate_c + 1])
    
  def forward(self, input):
    output = self.vgg_19(input)
    return output

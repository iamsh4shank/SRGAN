import cv2
from dataloader import DataLoader
import torch 
from torch import nn
import torchvision
import math

srgan_model = './checkpoint_srgan.pth.tar'
srgan_gen = torch.load(srgan_model)['generator'].to(device)

srgan_gen.eval()
model = srgan_gen

test_x = scipy.misc.imread('./test2.png', mode='RGB').astype(np.float)
test_x = torch.tensor(test_x).to(device)
test_x = test_x.view(1,3,test_x[2],test_x[3])
sr_imgs = model(test_x.float())
sr_imgs = (sr_imgs + 1.) / 2.
if sr_imgs.ndimension() == 3:
    sr_imgs = (sr_imgs - imagenet_mean) / imagenet_std
elif sr_imgs.ndimension() == 4:
    sr_imgs = (sr_imgs - imagenet_mean_cuda) / imagenet_std_cuda
A = sr_imgs.view(1,test_x[2],test_x[3],3).cpu().detach().numpy()

fig = plt.figure()
plt.imshow(A[0].astype(np.uint8))
fig.savefig("image1.png")
plt.close()
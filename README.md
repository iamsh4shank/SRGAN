# GalaxySuperResolution

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network Main target is to reconstruct a super-resolution image or high-resolution image by up-scaling low-resolution images such that texture detail in the reconstructed SR images is not lost.

Steps - 
1. It processes the HR images to get down-sampled LR(Low Resolution) images. Now it has both HR and LR images for the training data set. 

2. Pass the LR images through Generator which up-samples and gives SR(Super Resolution) images.

3. Then it uses a discriminator to distinguish the HR images and back-propagate the GAN loss to train the discriminator and the generator.

I implemented the paper - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)


<table>
    
<tr>
    <td><img height="250" src="https://i.imgur.com/d1dBZwr.png" /><br /><center><b>Low resolution Image</b></center>
    <td><img height="250" src="https://i.imgur.com/lZxGYcQ.png" /><br /><center><b>Generate HR image</b></center></td>
    </tr>

</table>

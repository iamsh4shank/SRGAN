# GalaxySuperResoulution

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network Main target is to reconstruct a super-resolution image or high-resolution image by up-scaling low-resolution images such that texture detail in the reconstructed SR images is not lost.

Steps - 
1. It processes the HR images to get down-sampled LR(Low Resolution) images. Now it has both HR and LR images for the training data set. 

2. Pass the LR images through Generator which up-samples and gives SR(Super Resolution) images.

3. Then it uses a discriminator to distinguish the HR images and back-propagate the GAN loss to train the discriminator and the generator.

I implemented the paper - [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)


<table>

<tr>
    <td><img height="250" src="https://github.com/iamsh4shank/ImageX/blob/main/screenshot/Screenshot%20from%202020-12-01%2002-15-02.png?raw=true"  /><br /><center><b>Low resolution Image</b></center>
    <td><img height="250" src="https://github.com/iamsh4shank/ImageX/blob/main/screenshot/Screenshot%20from%202020-12-01%2002-15-09.png?raw=true" /><br /><center><b>LR 2 image</b></center></td><td><img height="250" src="https://github.com/iamsh4shank/ImageX/blob/main/screenshot/Screenshot%20from%202020-12-01%2002-14-55.png?raw=true" /><br /><center><b>HR  generated Image</b></center></td> 
    </tr>

</table>
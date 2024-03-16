## Introduction
 This is a super resolution (SR) task using DL, the aim is to turn blurred images into clean images. The training images and task images came from the slices of PET_SIMPLE Image.
##  Division of training dataset and test datset
 The ratio of training dataset and test dataset is 8:2. The blurred low resolution images are the inputs and corresponding original clean images are the labels(target), the outputs are the reconstruct high resolution images.
## Model
 SRCNN[1] is a DL model for the SR task, the model was proposed by Dong,etc at 2014.
## Evaluation metrics
 PSIR and SSIM are the evaluation metrics,When evaluating on the test dataset, the code would output each images' PSNR and SSIM scores and output the low resolution image, reconstruct image, and original high resolution images of a random data in test dataset.Also,compute the ratio of good PSNR (above 50dB) and the ratio of good SSIM (above 0.75).
> Dong, C., Loy, C. C., He, K., & Tang, X. (2014). Learning a deep convolutional network for image super-resolution. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part IV 13 (pp. 184-199). Springer International Publishing.
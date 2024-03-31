import numpy as np
from noise import Noise
from enhancement import Histog
import matplotlib.pyplot as plt
import cv2

############## MAIN ################

# Image paths
IMAGE_PATH = './images/Image.jpg'
REFERENCE_IMAGE_PATH = './images/reference.jpg'

# Read images
# Read target image and normalized it
target_image = cv2.imread(IMAGE_PATH,0) / 255.0
# Read reference image
reference_image = cv2.imread(REFERENCE_IMAGE_PATH,0)

# Add impulsive noise
noisy_object = Noise()
noisy_image = noisy_object.impulsive(target_image,800)

# Remove noise
result_image_wo_noise = noisy_object.rem_impulsive_noise(noisy_image)
result_unsharp_image = noisy_object.rem_impulsive_noise_unsharp(noisy_image)

# return to 0-255 range
image = result_unsharp_image * 255
# Clip the values to the range [0, 255]
image = np.round(image) 
image = np.clip(image, 0, 255) 
# Convert the image to an 8-bit unsigned integer
image = image.astype(np.uint8) 

# Contrast enhancement

hist_object = Histog()
eq_img1 = hist_object.sqrt_contrast(image)
eq_img2 = hist_object.histogram_equalization(image)
eq_img3 = hist_object.linear_contrast(image)
hist_eq1,_ = hist_object.calc_histog_cdf(eq_img1)
hist_eq2,_ = hist_object.calc_histog_cdf(eq_img2)
hist_eq3,_ = hist_object.calc_histog_cdf(eq_img3)

hist,_ = hist_object.calc_histog_cdf(image)
ref_hist,_ = hist_object.calc_histog_cdf(reference_image)
spec_image = hist_object.spec(image,reference_image)
spec_hist,_ = hist_object.calc_histog_cdf(spec_image)


############## Visualization #############

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(target_image, cmap='gray')
axs[0,0].set_title('Target Image')
axs[0,1].imshow(noisy_image, cmap='gray')
axs[0,1].set_title('Noisy Image')
axs[1,0].imshow(result_image_wo_noise, cmap='gray')
axs[1,0].set_title('Image without noise')
axs[1,1].imshow(result_unsharp_image, cmap='gray')
axs[1,1].set_title('Image without noise and unsharp mask')
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(result_unsharp_image, cmap='gray')
axs[0,0].set_title('Image without noise and unsharp mask')
axs[0,1].imshow(eq_img2, cmap='gray')
axs[0,1].set_title('Histogram Equalization')
axs[1,0].bar(range(256), hist)
axs[1,0].set_title('Histogram of Image without noise and unsharp mask')
axs[1,1].bar(range(256), hist_eq2)
axs[1,1].set_title('Histogram of Histogram Equalization')
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(eq_img1, cmap='gray')
axs[0,0].set_title('Contrast Enhancement with square root')
axs[0,1].imshow(eq_img3, cmap='gray')
axs[0,1].set_title('Contrast Enhancement with linear contrast')
axs[1,0].bar(range(256), hist_eq1)
axs[1,0].set_title('Histogram of Contrast Enhancement with square root')
axs[1,1].bar(range(256), hist_eq3)
axs[1,1].set_title('Histogram of Contrast Enhancement with linear contrast')
plt.show()

fig, axs = plt.subplots(2,3)
axs[0,0].imshow(result_unsharp_image, cmap='gray')
axs[0,0].set_title('Image without noise and unsharp mask')
axs[0,1].imshow(reference_image, cmap='gray')
axs[0,1].set_title('Reference Image')
axs[0,2].imshow(spec_image, cmap='gray')
axs[0,2].set_title('Specified Image')
axs[1,0].bar(range(256), hist)
axs[1,0].set_title('Histogram of Image without noise and unsharp mask')
axs[1,1].bar(range(256), ref_hist)
axs[1,1].set_title('Histogram of Reference Image')
axs[1,2].bar(range(256), spec_hist)
axs[1,2].set_title('Histogram of Specified Image')
plt.show()




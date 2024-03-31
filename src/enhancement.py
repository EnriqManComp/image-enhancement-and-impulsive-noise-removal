import numpy as np

class Histog():

    def __init__(self):
        pass
    
    def histogram_equalization(self, image):
        """
            Method to enhance the contrast of an image using histogram equalization.
            Args:
                image: numpy array representing the image.
        
        """        
        hist = np.zeros(256, dtype=int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1

        # Compute the cumulative distribution function (CDF) of the histogram
        cdf = np.cumsum(hist)

        # Normalize the CDF to be between 0 and 1
        cdf_normalized = cdf / float(np.sum(hist))

        # Adjust the CDF using a gamma correction factor
        gamma = 1.6
        # Apply gamma correction to the normalized CDF 
        adjusted_cdf = cdf_normalized ** gamma

        # Scale the adjusted CDF to be between 0 and 255
        adjusted_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel_value = image[i, j]
                adjusted_pixel_value = int(adjusted_cdf[pixel_value] * 255)
                adjusted_image[i, j] = adjusted_pixel_value

        return adjusted_image
    

    def sqrt_contrast(self, image):
        """
            Sqrt contrast enhancement.
            Args:
                image: numpy array representing the image.
            Returns:
                adjusted_image: image with the contrast enhanced.            
        
        """
        
        height, width = image.shape
        
        min_val = float('inf')
        max_val = float('-inf')
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                if pixel_val < min_val:
                    min_val = pixel_val
                if pixel_val > max_val:
                    max_val = pixel_val

        diff = max_val - min_val
        # Avoid division by zero
        if diff == 0:
            diff = 0.0001
        
        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                # Apply the square root transformation
                adjusted_val = int(((np.sqrt(pixel_val) - min_val) * 255 / diff) ** 2)
                # Clip the values to the range [0, 255]
                adjusted_val = max(0, min(255, adjusted_val))
                # Assign the adjusted value to the pixel
                adjusted_image[i, j] = adjusted_val

        return adjusted_image

    def linear_contrast(self, image):
        """
            Linear contrast enhancement.
            Args:
                image: numpy array representing the image.
            Returns:
                adjusted_image: image with the contrast enhanced.            
        
        """
        
        height, width = image.shape
        
        min_val = float('inf')
        max_val = float('-inf')
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                if pixel_val < min_val:
                    min_val = pixel_val
                if pixel_val > max_val:
                    max_val = pixel_val

        diff = max_val - min_val

        if diff == 0:
            diff = 0.0001

        adjusted_image = np.zeros_like(image, dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                pixel_val = image[i, j]
                adjusted_val = ((pixel_val - min_val) * 255 / diff).astype(np.uint8)
                adjusted_image[i, j] = adjusted_val

        return adjusted_image

    def calc_histog_cdf(self,image):
        """
            Calculate the histogram and the cumulative distribution function (CDF) of an image.
            Args:
                image: numpy array representing the image.
            Returns:
                hist: histogram of the image.
                cdf: cumulative distribution function of the image.
        """
        
        hist = np.zeros(256, dtype=int)        
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                hist[image[i, j]] += 1
        
        cdf = np.cumsum(hist)        
        nj = (cdf - cdf.min()) * 255
        N = cdf.max() - cdf.min()
        cdf = nj / N        
        cdf = cdf.astype('uint8')
        return hist, cdf
    
    def spec(self,image,ref_image):
        """
            Global Image specification.
            Args:
                image: numpy array representing the image.
                ref_image: numpy array representing the reference image.
            Returns:
                img_new: image with the contrast enhanced.
        """
        
        flat_image = image.flatten()        
        _,cdf_ref_image = self.calc_histog_cdf(ref_image)
        
        img_new = cdf_ref_image[flat_image]        
        img_new = np.reshape(img_new,image.shape)

        return img_new


    
import numpy as np

class Noise():
    
    def __init__(self):
        pass

    def unsharp(self, px, EV_list, C):
        """
            Method to enhance the details of an image.
            Args:
                px: pixel value.
                EV_list: list of nearly value pixels.
                C: constant value.
            Returns:
                px: pixel value with the enhancement.

        """
        # Apply the unsharp mask formula to enhance the details
        mask = C*(px - np.median(EV_list))
        return px + mask    

    def impulsive(self,image,num_pixels):
        """
            Add impulsive noise to an image.
            Args:
                image: numpy array representing the image.
                num_pixels: number of pixels to be affected by the noise.
            Returns:
                noisy_image: image with impulsive noise.
        
        """

        # Image with impulsive noise
        noisy_image = np.copy(image)
        # Generate random coordinates for the pixels affected by the noise        
        x_coords = np.random.randint(0, image.shape[0], size=num_pixels)
        y_coords = np.random.randint(0, image.shape[1], size=num_pixels)
        # Asign values of salt or pepper to the selected pixels        
        for i in range(len(x_coords)):
            # Generate a random number to decide if the pixel will be salt or pepper
            t = np.random.uniform(0, 1)            
            if t <= 0.5:
                noisy_image[x_coords[i],y_coords[i]] = 0.  # Pepper
            else:
                noisy_image[x_coords[i],y_coords[i]] = 1.  # Salt
        return noisy_image

    def EV(self,kernel,epsilon):
        """
            Method of remove noise based in nearly values.
            Args:
                kernel: numpy array representing the kernel.
                epsilon: threshold value.

            Returns:
                EV_list: list of nearly value pixels.
        
        """
        # EV neighbor pixels
        EV_list = []
        central_pos = kernel.shape[0] // 2
        central_px = kernel[central_pos, central_pos]
        inferior_limit = central_px - epsilon
        superior_limit = central_px + epsilon
        for k in range(kernel.shape[0]):
            for l in range(kernel.shape[1]):
                if k != central_pos and l != central_pos:
                    # Check if the pixel is in the range
                    if kernel[k,l] >= inferior_limit and kernel[k,l] <= superior_limit:
                        EV_list.append(kernel[k,l])
        return EV_list
    
    def rem_impulsive_noise(self,noisy_image):
        """
            Remove impulsive noise from an image with a local kernel.
            Args:
                noisy_image: numpy array representing the image with noise.
            Returns:
                image_copy: image without noise.
        
        """
        # Copy of the noisy_image
        image_copy = np.copy(noisy_image)
        # Pass a 5x5 Kernel over the image        
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    kernel = noisy_image[n-2:n+3,m-2:m+3].copy()
                    # Get EV neighbor pixels
                    EV_list = self.EV(kernel,( 20 / 255.0 ))
                    # Process the EV list if it is less than 3 elements                    
                    if len(EV_list) < 3:
                        # Extract 8 near neighbors
                        neighbor_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1], \
                                           image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        # Sort the values
                        neighbor_values.sort()
                        # Replace the central pixel with median value
                        image_copy[n,m] = np.median(neighbor_values)                        
        return image_copy
    

    def rem_impulsive_noise_unsharp(self,noisy_image):
        """
            Remove impulsive noise from an image with a local kernel.
            Perform unsharp masking to enhance the details.

            Args:
                noisy_image: numpy array representing the image with noise.
            Returns:
                image_copy: image without noise.
        
        """
        # Copy of the noisy_image
        image_copy = noisy_image.copy()
        # Pass a 5x5 Kernel over the image
        for i in range(1):
            for n in range(2,image_copy.shape[0]-2,1):
                for m in range(2,image_copy.shape[1]-2,1):
                    # Get the 5x5 kernel
                    kernel = np.copy(image_copy[n-2:n+3,m-2:m+3])
                    # Get EV neighbor pixels
                    EV_list = self.EV(kernel,(20/255.))
                    # Sort the EV list
                    EV_list.sort()
                    if len(EV_list) < 3:
                        # Extract 8 near neighbors
                        neighbor_values = [image_copy[n-1,m],image_copy[n+1,m],image_copy[n,m-1],image_copy[n,m+1], \
                                            image_copy[n-1,m-1],image_copy[n-1,m+1],image_copy[n+1,m-1],image_copy[n+1,m+1]]                                       
                        # Sort the values
                        neighbor_values.sort()
                        # Replace the central pixel with median value
                        image_copy[n,m] = np.median(neighbor_values)                                                
                    else:
                        # If not detected noise, apply unsharp masking
                        central_pos = kernel.shape[0] // 2
                        # Get the central pixel value
                        central_px = kernel[central_pos, central_pos]
                        # Apply the unsharp mask
                        unsharp_kernel = self.unsharp(central_px, EV_list, 1.)                                                                   
                        # Replace the central pixel with the unsharp mask value
                        image_copy[n,m] = unsharp_kernel                                            
        return image_copy



    
    
    
    
    

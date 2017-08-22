import cv2
import numpy as np

def convolve(input, kernel):
    # Create the output matrix
    output = np.zeros(input.shape)
    
    # Verify the convolution type chrome or BGR
    if(input.ndim == 2):
        convolution(input, kernel, output, -1)
    else:
        for i in range(3):
            convolution(input[:,:,i], kernel, output, i)

    return output

def convolution(input, kernel, output, channel):
    # Flips kernel to convolve
    kernel = np.flip(np.flip(kernel,0),1)

    # Get image information
    heightI, widthI = input.shape[:2]    
    heightK, widthK = kernel.shape[:2]

    # Calculate the center, It will be used as difference between input and kernel
    diff = (int) (heightK/2)
    
    # Create border on image before convolving (using difference between kernel and input)
    input = cv2.copyMakeBorder(input, diff, diff, diff, diff, cv2.BORDER_CONSTANT, 0)

    # Slides kernel over the new bordered image
    for i in range(diff, heightI + diff):
        for j in np.arange(diff, widthI + diff):
            value = np.sum(input[i-diff:i+diff +1, j-diff:j+diff+1] * kernel)
            
            if(channel == -1):
                output[i-diff,j-diff] = value
            else:
                output[i-diff,j-diff,channel] = value


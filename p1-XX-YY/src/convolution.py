import numpy as np
import copy as cp

def convolve(input, mask):
    input = np.array(input)
    mask = np.array(mask)

    output = cp.copy(input)
    np.array(output)
    
    if(input.ndim == 2):
        convolution(input, mask, output, -1)
    else:
        for i in range(2):
            convolution(input[:,:,i], mask, output, i)

    return output

def convolution(input, mask, output, channel):   
    mask = np.flip(np.flip(mask,0),1)

    heightI, widthI = input.shape    
    heightM, widthM = mask.shape
    c = (int) (heightM/2)
    
    for i, row in enumerate(input):
        for j, col in enumerate(row):
            value = 0
            for x in range(heightM):
                for y in range(widthM):
                    indexI = i+(x-c)
                    indexJ = j+(y-c)
                    
                    if(indexI >= 0 and indexI < heightI and indexJ >= 0 and indexJ < widthI):
                        value = value + input[indexI, indexJ] * mask[x,y]
            if(channel == -1):
                output[i,j] = value
            else:
                output[i,j,channel] = value


import cv2
import numpy as np
import heapq

def clearValues(matrix, perc, min):
    # Create 1D array to find min
    array = matrix.flatten()

    # Find nth value depending on %
    nth = int((array.shape[0]*perc)/100)

    # In decresing order change values
    if min == "desc":
        array = array * -1
        matrix = matrix * -1

    # Find nth min element
    cut = heapq.nsmallest(nth, array)[-1]

    # Zeroing values
    matrix[matrix > cut] = 0

    # In decresing order change values back
    if min == "desc":
        matrix = matrix * -1

    return matrix

def transform(img):
    # Uses discrete fourier transformation
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    # Shifts image to center
    dft_center = np.fft.fftshift(dft)

    # Gets magnitude and phase
    magnitude = 20*np.log(cv2.magnitude(dft_center[:, :, 0], dft_center[:, :, 1]))
    phase = 40*np.log(cv2.phase(dft_center[:, :, 0], dft_center[:, :, 1], True))

    return magnitude, phase

def reconstruct(magnitude, phase, type, perc, order):
    # Reconstruct values of magnitude and phase
    magnitude = np.exp(magnitude/20)
    phase = np.exp(phase/40)

    # Apply the porcentage into the frequency
    if(type == "phase"):
        phase = clearValues(phase, perc, order)
    else:
        magnitude = clearValues(magnitude, perc, order)

    # Creates comples function
    func = magnitude * np.exp(1j*phase)

    # Executes inverse fourier transformation
    idf = np.fft.ifft2(func)

    # Create new image
    image_back = np.abs(idf)

    # Rotate image because of shift operation
    height, width = image_back.shape
    center = (width/2, height/2)
    rotation = cv2.getRotationMatrix2D(center, 180, 1)
    image_back = cv2.warpAffine(image_back, rotation, (width, height))

    return image_back
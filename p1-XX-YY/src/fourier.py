import cv2
import numpy as np
import heapq

def clearValues(matrix, perc, order):
    # Create 1D array to find min
    array = np.trim_zeros(matrix.flatten())

    if perc != -1:
        # Find nth value depending on %
        nth = int((array.shape[0]*perc)/100)
    else:
        nth = 1;

    # In decresing order change values
    if order == "desc":
        array = array * -1
        matrix = matrix * -1

    # Find nth min element
    cut = heapq.nsmallest(nth, array)[-1]

    # Zeroing values
    matrix[matrix > cut] = 0

    # In decresing order change values back
    if order == "desc":
        matrix = matrix * -1

    return matrix

def transform(img, visualize):
    # Uses discrete fourier transformation
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    # Shifts image to center
    dft_center = np.fft.fftshift(dft)

    # Gets magnitude and phase
    if visualize:
        # Adjust values to visualization
        magnitude = 20*np.log(cv2.magnitude(dft_center[:, :, 0], dft_center[:, :, 1]))
        phase = 40*np.log(cv2.phase(dft_center[:, :, 0], dft_center[:, :, 1], True))
    else:
        magnitude = cv2.magnitude(dft_center[:, :, 0], dft_center[:, :, 1])
        phase = cv2.phase(dft_center[:, :, 0], dft_center[:, :, 1], True)

    return magnitude, phase

def invertTranform(func):
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

def reconstruct(magnitude, phase, type, perc, order, visualize):
    # If transformed for visualization
    if visualize:
        # Reconstruct values of magnitude and phase
        magnitude = np.exp(magnitude/20)
        phase = np.exp(phase/40)

    # Apply the porcentage into the frequency
    if type == "phase":
        phase = clearValues(phase, perc, order)
    elif type == "magnitude":
        magnitude = clearValues(magnitude, perc, order)

    # Creates complex function
    func = magnitude * np.exp(1j*phase)

    # Makes inverse fourier transform
    image_back = invertTranform(func)

    return image_back

def blend_frequencies(img1, img2, mask):
    img1 = img1 * (mask/255)
    img2 = img2 * (1 - mask/255)

    # Tranform images to frequency domain
    img1_mag, img1_phase = transform(img1, False)
    img2_mag, img2_phase = transform(img2, False)

    # Creates complex function of frequencies
    img1_func = img1_mag * np.exp(1j*img1_phase)
    img2_func = img2_mag * np.exp(1j*img2_phase)

    # Sum both complex functions to blend images
    img_func = img1_func + img2_func

    # Executes inverse fourier transform
    img = invertTranform(img_func)

    return img

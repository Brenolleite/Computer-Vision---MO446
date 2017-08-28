import cv2
import numpy as np

def transform(img):
    # Uses discrete fourier transformation
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    # Shifts image to center
    dft_center = np.fft.fftshift(dft)

    # Gets magnitude and phase
    magnitude = 20*np.log(cv2.magnitude(dft_center[:, :, 0], dft_center[:, :, 1]))
    phase = 40*np.log(cv2.phase(dft_center[:, :, 0], dft_center[:, :, 1], True))

    return magnitude, phase

def reconstruct(magnitude, phase, perc):
    # Reconstruct values of magnitude and phase
    magnitude = np.exp(magnitude/20)
    phase = np.exp(phase/40)

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

# The type of file (0) is necessary in this case
input = cv2.imread('../input/p1-1-0.png', 0)

magnitude, phase = transform(input)

cv2.imwrite('../output/phase.png', phase)
cv2.imwrite('../output/magnitude.png', magnitude)

phase_back = reconstruct(magnitude, phase, 100)

cv2.imwrite('../output/back.png', phase_back)
#cv2.imwrite('../output/magnitude_back.png', magnitude_back)
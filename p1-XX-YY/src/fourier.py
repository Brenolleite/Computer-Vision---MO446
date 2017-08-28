import cv2
import numpy as np

def transform(img):
    # Uses discrete fourier transformation
    dft = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)

    # Shifts image to center
    dft_center = np.fft.fftshift(dft)

    # Gets magnitude and phase
    magnitude = 20*np.log(cv2.magnitude(dft_center[:, :, 0], dft_center[:, :, 1]))
    phase = 20*np.log(cv2.phase(dft_center[:, :, 0], dft_center[:, :, 1], True))

    return magnitude, phase

def reconstruct(magnitude, phase, perc):
    # Reshifts the images out off center
    magnitude = np.fft.ifftshift(magnitude)
    phase = np.fft.ifftshift(phase)

    # Performs inverse discrete fourier transformation
    idft_magnitude = cv2.idft(magnitude)
    idft_phase = cv2.idft(phase)

    # Performs inverse magnitude/phase process
    img_magnitude = cv2.magnitude(idft_magnitude[:, :, 0], idft_magnitude[:, :, 1])
    img_phase = cv2.phase(idft_phase[:, :, 0], idft_phase[:, :, 1])

    return img_magnitude, img_phase

# The type of file (0) is necessary in this case
input = cv2.imread('../input/p1-1-0.png', 0)

magnitude, phase = transform(input)

cv2.imwrite('../output/phase.png', phase)
cv2.imwrite('../output/magnitude.png', magnitude)

magnitude_back, phase_back = reconstruct(magnitude, phase, 100)

cv2.imwrite('../output/phase_back.png', phase_back)
cv2.imwrite('../output/magnitude_back.png', magnitude_back)
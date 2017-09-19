import numpy as np
import copy as cp
import cv2
import math

octave_number = 4
interval_number = 5
init_sigma = math.sqrt(2)
gaussian_size = (7, 7)

# Return the descriptors and keypoints found in the image
def sift(img):
    oc_list, sigma_m = DoG(img, octave_number, interval_number)
    detectExtreme(oc_list)

# Create the octaves, scales and Difference of Gaussian matrix
def DoG(image, o_num, i_num):
    img = cp.copy(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussian_size, 0.5)
    gray = cv2.pyrUp(gray)

    # Octave matrix
    oc_list = []
    int_list = []
    int_list.append(gray)
    oc_list.append(int_list)

    # DoG matrix
    dog_list = []
    rowDog_list = []
    dog_list.append(rowDog_list)

    # Sigma matrix
    sigma_m = []
    sigma_list = []
    sigma_m.append(sigma_list)

    # Create octaves
    for i in range(o_num):

        sigma = init_sigma

        # Create scales
        for k in range(1, i_num + 3):
            sigma_f = math.sqrt(math.pow(2.0, 2.0 / i_num) - 1) * sigma
            sigma = math.pow(2.0, 1.0 / i_num) * sigma
            sigma_m[i].append(sigma * 0.5 * math.pow(2.0, i))

            aux = cv2.GaussianBlur(oc_list[i][k - 1], gaussian_size, sigma_f)
            oc_list[i].append(aux)

            oc_list[i][k - 1] = oc_list[i][k - 1].astype(np.int16)
            dog_aux = oc_list[i][k - 1] - oc_list[i][k]
            dog_list[i].append(dog_aux)

        # Create DoG
        if(i < o_num - 1):
            int_list = []
            aux = cv2.pyrDown(oc_list[i][0])
            int_list.append(aux)
            oc_list.append(int_list)

            sigma_m.append(sigma_list)

            dog_list.append(rowDog_list)

    return (oc_list, sigma_m)

# Find the maxima and minima point in the DoG
def detectExtreme(oc_list):
    # Number of points found
    key_num = 0
    # Number of points found and rejected after further testing
    key_rejected = 0

    # Matrix of maximas and minimas
    m_extrema = []
    e_list = []
    m_extrema.append(e_list)

    # Run over a octave
    for i in range(octave_number):
        scale = math.pow(2.0, i)

        # Run over the DoG in this octave
        for k in range(1, interval_number + 1):
            print(i, k)

            # The three DoGs that will be searching on
            mid = oc_list[i][k]
            above = oc_list[i][k + 1]
            below = oc_list[i][k - 1]

            # Running over all the DoG comparing if point(i, k) is the maxima / minima
            for l in range(1, oc_list[i][k].shape[0] - 1):
                for m in range(1, oc_list[i][k].shape[1] - 1):
                    justSet = False

                    c_pixel = cv2.GetReal2D(mid, m, l)

                    if((c_pixel > cv2.GetReal2D(mid, m - 1, l)) and
                        (c_pixel > cv2.GetReal2D(mid, m + 1, l)) and
                        (c_pixel > cv2.GetReal2D(min, m, l - 1)) and
                        (c_pixel > cv2.GetReal2D(min, m, l + 1)) and
                        (c_pixel > cv2.GetReal2D(min, m - 1, l - 1)) and
                        (c_pixel > cv2.GetReal2D(min, m - 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(min, m + 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(min, m + 1, l - 1)) and
                        (c_pixel > cv2.GetReal2D(above, m, l)) and
                        (c_pixel > cv2.GetReal2D(above, m - 1, l)) and
                        (c_pixel > cv2.GetReal2D(above, m + 1, l)) and
                        (c_pixel > cv2.GetReal2D(above, m, l - 1)) and
                        (c_pixel > cv2.GetReal2D(above, m, l + 1)) and
                        (c_pixel > cv2.GetReal2D(above, m - 1, l - 1)) and
                        (c_pixel > cv2.GetReal2D(above, m - 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(above, m + 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(above, m + 1, l - 1)) and
                        (c_pixel > cv2.GetReal2D(below, m, l)) and
                        (c_pixel > cv2.GetReal2D(below, m - 1, l)) and
                        (c_pixel > cv2.GetReal2D(below, m + 1, l)) and
                        (c_pixel > cv2.GetReal2D(below, m, l - 1)) and
                        (c_pixel > cv2.GetReal2D(below, m, l + 1)) and
                        (c_pixel > cv2.GetReal2D(below, m - 1, l - 1)) and
                        (c_pixel > cv2.GetReal2D(below, m - 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(below, m + 1, l + 1)) and
                        (c_pixel > cv2.GetReal2D(below, m + 1, l - 1))):
                            # Is maxima
                            cv2.SetReal2D(m_extrema[i][k - 1], l, m, 255)
                            key_num += 1
                            justSet = True
                    elif((c_pixel < cv2.GetReal2D(mid, m - 1, l)) and
                        (c_pixel < cv2.GetReal2D(mid, m + 1, l)) and
                        (c_pixel < cv2.GetReal2D(min, m, l - 1)) and
                        (c_pixel < cv2.GetReal2D(min, m, l + 1)) and
                        (c_pixel < cv2.GetReal2D(min, m - 1, l - 1)) and
                        (c_pixel < cv2.GetReal2D(min, m - 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(min, m + 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(min, m + 1, l - 1)) and
                        (c_pixel < cv2.GetReal2D(above, m, l)) and
                        (c_pixel < cv2.GetReal2D(above, m - 1, l)) and
                        (c_pixel < cv2.GetReal2D(above, m + 1, l)) and
                        (c_pixel < cv2.GetReal2D(above, m, l - 1)) and
                        (c_pixel < cv2.GetReal2D(above, m, l + 1)) and
                        (c_pixel < cv2.GetReal2D(above, m - 1, l - 1)) and
                        (c_pixel < cv2.GetReal2D(above, m - 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(above, m + 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(above, m + 1, l - 1)) and
                        (c_pixel < cv2.GetReal2D(below, m, l)) and
                        (c_pixel < cv2.GetReal2D(below, m - 1, l)) and
                        (c_pixel < cv2.GetReal2D(below, m + 1, l)) and
                        (c_pixel < cv2.GetReal2D(below, m, l - 1)) and
                        (c_pixel < cv2.GetReal2D(below, m, l + 1)) and
                        (c_pixel < cv2.GetReal2D(below, m - 1, l - 1)) and
                        (c_pixel < cv2.GetReal2D(below, m - 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(below, m + 1, l + 1)) and
                        (c_pixel < cv2.GetReal2D(below, m + 1, l - 1))):
                            # Is minima
                            cv2.SetReal2D(m_extrema[i][k - 1], m, l, 255)
                            key_num += 1
                            justSet = True

                    # Further testing if contrast is above threshold
                    if(justSet and abs(cv2.getReal2D(mid, m, l)) < contrast_threshold):
                        cv2.setReal2D(m_extrema[i][k - 1], m, l, 0)
                        key_num -= 1
                        key_rejected += 1
                        justSet = False

input = cv2.imread('../input/img2.png')
sift(input)

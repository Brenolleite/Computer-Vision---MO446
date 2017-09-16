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
    buildScaleSpace(img, octave_number, interval_number)

def buildScaleSpace(image, o_num, i_num):
    print("Building Scale Space")

    img = cp.copy(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, gaussian_size, 0.5)
    gray = cv2.pyrUp(gray)

    oc_list = []
    int_list = []
    int_list.append(gray)
    oc_list.append(int_list)

    dog_list = []
    rowDog_list = []
    dog_list.append(rowDog_list)

    sigma_m = []
    sigma_list = []
    sigma_m.append(sigma_list)

    for i in range(o_num):

        sigma = init_sigma

        for k in range(1, i_num + 3):
            print(i, k)
            sigma_f = math.sqrt(math.pow(2.0, 2.0 / i_num) - 1) * sigma
            sigma = math.pow(2.0, 1.0 / i_num) * sigma
            sigma_m[i].append(sigma * 0.5 * math.pow(2.0, i))

            aux = cv2.GaussianBlur(oc_list[i][k - 1], gaussian_size, sigma_f)
            oc_list[i].append(aux)

            oc_list[i][k - 1] = oc_list[i][k - 1].astype(np.int16)
            dog_aux = oc_list[i][k - 1] - oc_list[i][k]
            dog_list[i].append(dog_aux)

            print("Intensidade Max: ", dog_aux.max())

            if k == 3:
                print(dog_aux)

            cv2.imwrite('debug/DoG-{}-{}.png'.format(i, k - 1), dog_list[i][k - 1])
            cv2.imwrite('debug/octave-{}-{}.png'.format(i, k - 1), oc_list[i][k - 1])

        if(i < o_num - 1):
            int_list = []
            aux = cv2.pyrDown(oc_list[i][0])
            int_list.append(aux)
            oc_list.append(int_list)

            sigma_m.append(sigma_list)

            dog_list.append(rowDog_list)

    return (oc_list, sigma_m)

input = cv2.imread('input/img2.png')
sift(input)

import numpy as np

def transform(img, A):
    height, width, channels = img.shape
    final_img = np.zeros(img.shape)
    for j in range(height):
        for i in range(width):
            x = int(np.floor(A[0][0] * i + A[1][0] * j + A[2][0]))
            y = int(np.floor(A[0][1] * i + A[1][1] * j + A[2][1]))
            if y >= 0 and y < height and x >= 0 and x < width:
                final_img[y, x, :] = img[j, i, :]

    return final_img



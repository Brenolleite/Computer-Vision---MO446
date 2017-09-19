import numpy as np
import cv2
import copy as cp

MIN_MATCH_COUNT = 10

img1 = cv2.imread('input/img1.png')          # queryImage
img2 = cv2.imread('input/img2.png') # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

print("KeyPoints: ", len(kp1), len(kp2))
print("Descriptors: ", len(des1), len(des2))

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#    print("dest: ", type(dst_pts))
#    print("dest: ", len(dst_pts))
#    print("dest: ", type(dst_pts[0]))
#    print("dest: ", len(dst_pts[0]))
#    print("src:\n", src_pts)
#    print("dst:\n", dst_pts)

    M = cv2.estimateRigidTransform(src_pts, dst_pts, True)

#    print("M Type: ", type(M))
#    print("M Size: ", len(M))
#    print("M 0: ", len(M[0]))
#    print("M 1: ", len(M[1]))
#    print("M 2: ", len(M[2]))
#    print("mask type: ", type(mask))
#    print("mask size: ", len(mask))
#    print("mask 0: ", type(mask[0]))

    print("M:\n", M)

    # matchesMask = mask.ravel().tolist()

    h, w, y = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
#    dst = cv2.perspectiveTransform(pts,M)

#    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    o = 0
#    matchesMask = None
# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                   singlePointColor = None,
#                   matchesMask = matchesMask, # draw only inliers
#                   flags = 2)

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
for i in range(img1.shape[0]):
    for k in range(img1.shape[1]):
        A = np.matrix([i, k])
        print("DOT: ", np.dot(A, M))
        #x, y = np.dot(A, M)
        #print(i, k)
        #print(x, y)
        #print("\n")

im_dst = cv2.warpAffine(img1, M, (img1.shape[0], img1.shape[1]))

#cv2.imwrite('debug/RANSAC_test.png', img3)
cv2.imwrite('debug/TRANSFORM_test.png', im_dst)

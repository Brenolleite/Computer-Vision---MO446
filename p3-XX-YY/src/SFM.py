import numpy as np
import opencv
from numpy.linalg import lstsq
from numpy.linalg import inv
from numpy.linalg import cholesky
import KLT as motion

def create_rowG(a, b):
    # Creating system for matrix G
    i1 = a[0]*b[0]
    i2 = a[0]*b[1] + a[1]*b[0]
    i3 = a[0]*b[2] + a[2]*b[0]
    i4 = a[1]*b[1]
    i5 = a[1]*b[2] + a[2]*b[1]
    i6 = a[2]*b[2]

    return [i1, i2, i3, i4, i5, i6]

def create_GC(M):
    G = []
    C = []

    # Creating 2f matrix G and C
    for i in range(len(M)):
        G.append(create_rowG(M[i], M[i]))
        C.append([1])

    # Creating f matrix G and C
    f = int(len(M)/2)
    for i in range(f):
        G.append(create_rowG(M[i], M[i+f]))
        C.append([0])

    return np.array(G), np.array(C)

def sfm(kps):
    W = []
    X = []
    Y = []

    # Creating matrix X and Y to create W
    for i in range(len(kps)):
        rowx = []
        rowy = []
        for j in range(len(kps[i])):
             rowx.append(kps[i,j,0])
             rowy.append(kps[i,j,1])

        X.append([rowx])
        Y.append([rowy])

    # Building matrix W
    W = np.concatenate((np.array(X), np.array(Y)), axis=0).squeeze()

    # SVD factorization
    U, S, V = np.linalg.svd(W, full_matrices=True)

    # Transform to ndarray
    U = np.array(U)
    S = np.array(S)
    V = np.array(V)

    # Getting 3 points from SVD
    M = U[:, 0:3]
    S = np.dot(np.diag(S[0:3]), V[0:3,:])

    # Creating Gl = C
    G, C = create_GC(M)

    # Solving Gl = C, to get l
    Gt = np.transpose(G)
    l = np.dot(np.dot(inv(np.dot(Gt, G)), Gt), C)

    # Creating L from l
    L = [[l[0,0], l[1,0], l[2,0]], [l[1,0], l[3,0], l[4,0]], [l[2,0], l[4,0], l[5,0]]]

    # Finding A using cholesky
    A = cholesky(L)

    # Getting real matrixes M and S
    M = np.dot(M, A)
    S = np.dot(inv(A), S)

    # Creating points to meshlab
    points = np.transpose(S)

    # Creating colors to points to object
    colors = []
    cam_colors = []
    cam_points = []
    for i in range(len(points)):
        colors.append([102, 255, 102])

    # Creating color to the camera position
    f = int(len(M)/2)
    for i in range(f):
        # Crossing X and Y to get perpendicular vector (camera positions)
        cam_points = np.append(cam_points, (np.cross(M[i], M[i+f])))

        # Chosing black color to camera position
        cam_colors.append([0, 0, 0])

    return points,  np.array(colors), cam_points,  np.array(cam_colors)
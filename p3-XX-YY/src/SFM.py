import numpy as np
import opencv
from numpy.linalg import lstsq
from numpy.linalg import inv
from numpy.linalg import cholesky
import meshlab as ml

def create_rowG(a, b):
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

    for i in range(len(M)):
        G.append(create_rowG(M[i], M[i]))
        C.append([1])


    f = int(len(M)/2)
    for i in range(f):
        G.append(create_rowG(M[i], M[i+f]))
        C.append([0])

    return np.array(G), np.array(C)

def sfm(kps):
    W = []
    X = []
    Y = []
    for i in range(len(kps)):
        rowx = []
        rowy = []
        for j in range(len(kps[i])):
             rowx.append(kps[i,j,0])
             rowy.append(kps[i,j,1])

        X.append([rowx])
        Y.append([rowy])

    W = np.concatenate((np.array(X), np.array(Y)), axis=0).squeeze()

    U, S, V = np.linalg.svd(W, full_matrices=True)

    U = np.array(U)
    S = np.array(S)
    V = np.array(V)

    M = U[:, 0:3]
    S = np.dot(np.diag(S[0:3]), V[0:3,:])

    G, C = create_GC(M)

    Gt = np.transpose(G)
    l = np.dot(np.dot(inv(np.dot(Gt, G)), Gt), C)

    L = [[l[0,0], l[1,0], l[2,0]], [l[1,0], l[3,0], l[4,0]], [l[2,0], l[4,0], l[5,0]]]

    A = cholesky(L)

    M = np.dot(M, A)
    S = np.dot(inv(A), S)

    St = np.transpose(S)

    colors = []
    for i in range(len(St)):
        colors.append([13, 94, 1])

    ml.write_ply('../output/teste.ply', St, np.array(colors))

video_path = '../input/teste2.mp4'
kps = opencv.KLT(video_path)
sfm(kps)
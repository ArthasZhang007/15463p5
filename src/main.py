import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.colors import LightSource
import math 
import scipy 
import cp_hw2 as helper
import cp_hw5


def convert(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def readimage(path):
    return cv2.imread(path).astype(np.float32) / 255.0

def showimage(image):
    plt.imshow(convert(image))
    plt.show()

def showimage_raw(image):
    plt.imshow(image)
    plt.show()



def writeimage(file, image):
    return cv2.imwrite('./data/output/' + file, (np.clip(image,0,1) * 255).astype(np.uint8))

def linearize(C):
    return (C < 0.0404482)/12.92 + (C > 0.0404482) * np.power((C+0.055)/1.055, 2.4)   

def getAN(B):
    rows, cols = B.shape[0], B.shape[1]
    vh = np.transpose(np.reshape(B, (rows * cols, 3)))
    A = np.linalg.norm(vh, axis = 0)
    N = vh/A
    A = np.reshape(A, (rows, cols))
    N = np.reshape(np.transpose(N), (rows, cols, 3))
    return A,N

def displayb(b):
    A,N = getAN(b)
    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(A * 10, cmap = 'gray')
    axarr[1].imshow((N+1)/2)
    plt.show() 

def displayZ(Z):
    # Z is an HxW array of surface depths
    H, W = Z.shape
    X, Y = np.meshgrid(np.arange(0,W), np.arange(0,H))

    ls = LightSource()
    color_shade = ls.shade(Z, plt.cm.gray)

    # set 3D figure
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, facecolors=color_shade,
                       rstride=1, cstride=1)
    plt.axis('off')
    plt.show()



def loadIstack():
    Istack = list()

    rows = 0
    cols = 0
    for i in range(7):
        I = readimage("../data/input_{:d}.tif".format(i+1));
        rows, cols = I.shape[0], I.shape[1]
        I = helper.lRGB2XYZ(convert(I))
        I_f = I[:, :, 1].flatten()
        
        Istack.append(I_f)
    I = np.vstack(np.array(Istack))
    return I, rows, cols

def uncali(I, rows, cols):
    u,s,vh = np.linalg.svd(I, full_matrices=False)
    u,s,vh = u[:,:3],s[:3],vh[:3,:]

    #print(s, np.sqrt(s))
    s = np.sqrt(s)
    u = np.transpose(u)
    # print(u)
    

    for i in range(3):
        u[i] = u[i] * s[i]
        vh[i] = vh[i] * s[i]
    
    # print(vh)
    # Q = np.array(
    #     [
    #         [1,0.5,0.5],
    #         [0.25,1,0.25],
    #         [0.5,0.5,1]
    #     ]
    # )
    # vh = Q @ vh
    # print(vh.shape)

    B = np.reshape(np.transpose(vh), (rows, cols, 3))
    displayb(B)
    return B
    
def enforce(b):
    sig = 11
    c_list = list()
    for c in range(3):
        s = b[:,:,c]
        c_list.append(scipy.ndimage.gaussian_filter(s, sig))
    B_G = np.dstack(c_list)
    d = np.gradient(B_G)
    dx,dy = d[1],d[0]
    
    def g(I, x):
        return I[:,:, x-1]
    
    A_1 = g(b,1) * g(dx, 2) - g(b,2) * g(dx, 1)
    A_2 = g(b,1) * g(dx, 3) - g(b,3) * g(dx, 1)
    A_3 = g(b,2) * g(dx, 3) - g(b,3) * g(dx, 2)
    A_4 =-g(b,1) * g(dy, 2) + g(b,2) * g(dy, 1)
    A_5 =-g(b,1) * g(dy, 3) + g(b,3) * g(dy, 1)
    A_6 =-g(b,2) * g(dy, 3) + g(b,3) * g(dy, 2)
    A = np.dstack([A_1.flatten(),
                   A_2.flatten(),
                   A_3.flatten(),
                   A_4.flatten(),
                   A_5.flatten(),
                   A_6.flatten()])
    A = A[0]
    u,s,vh = np.linalg.svd(A, full_matrices=False)
    x = -vh[-1]
    # print(A @ x)
    delta = np.array([
        [-x[3-1], x[6-1], 1],
        [x[2-1], -x[5-1], 0],
        [-x[1-1], x[4-1], 0]
    ])
    inv = np.linalg.inv(delta)
    flip = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            b[i][j] = flip @ inv @ b[i][j]
    return b
    #displayb(b)



def integrate(B):
    u = 1
    v = 1
    l = 1
    G = np.array([
        [1,0,0],
        [0,1,0],
        [u,v,l]
    ])
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = G @ B[i][j]
    # displayb(B)
    A,N = getAN(B)
    displayb(B)
    x = N[:,:,0]
    y = N[:,:,1]
    z = N[:,:,2]
    eps = 1.5
    # print(np.mean(z))
    dx = -x / (z + eps)
    dy = -y / (z + eps)
    #I = cp_hw5.integrate_poisson(dx, dy)
    
    I = cp_hw5.integrate_frankot(dx, dy)
    # plt.imshow(1 - I, cmap = 'gray')
    # plt.show()
    displayZ(I)



def main():
    I, rows, cols = loadIstack()
    
    # part A
    # B = uncali(I, rows, cols)
    # part B
    # B = enforce(B)
    # part D

    # part C
    # integrate(B)

main()
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import math 
import scipy 
import cp_hw2 as helper



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

def subslice(I, x, y):
    return I[x::16, y::16]

def linearize(C):
    return (C < 0.0404482)/12.92 + (C > 0.0404482) * np.power((C+0.055)/1.055, 2.4)   

def loadIstack():
    Istack = list()
    for i in range(7):
        I = readimage("../data/input_{:d}.tif".format(i+1));
        I = helper.lRGB2XYZ(convert(I))
        
        I_f = I[:, :, 1].flatten()
        #luminance channel

        # print(I.shape, I_f.shape)
        Istack.append(I_f)
    I = np.vstack(np.array(Istack))
    u,s,vh = np.linalg.svd(I, full_matrices=False)
    u,s,vh = u[:,0:3],s[0:3],vh[0:3,:]
    print(np.diag(s))

    #print(s)
    print(u.shape, s.shape, vh.shape)
    

def main():
    loadIstack()

main()
'''
utils.py
Created on 2012/12/05
@author: du
'''

import numpy as np
import cv2
import math

def convImgMat(img, mat_type="cv"):
    shape = img.shape
    img = np.asanyarray(img, dtype=float)
    if mat_type == "opencv" and len(shape) == 1:
        size = int(math.sqrt(img.size))
        img = np.array(img.reshape((size, size)) * 255.0, dtype=int)
    elif mat_type == "numpy" and len(shape) == 2:       
        img = img.flatten / 255.0
    else:
        raise ValueError("wrong format")
    return img

def toGray(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    return gray_img
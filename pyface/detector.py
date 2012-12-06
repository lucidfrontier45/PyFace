'''
detector.py
Created on 2012/12/05
@author: du
'''

import cv2
from cv2 import cv
import numpy as np
from .utils import toGray

#haar_detector = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
#lbp_detector = cv2.CascadeClassifier("data/lbpcascades/lbpcascade_frontalface.xml")

def detectFaces(gray_img, detector, flags=cv.CV_HAAR_SCALE_IMAGE, min_size=(64, 64)):
    rects = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3,
            minSize=min_size, flags=flags)
    faces = []
    for x, y, w, h in rects:
        face = np.array(gray_img[y:y+h, x:x+w])
        face = cv2.resize(face, min_size, interpolation=cv.CV_INTER_AREA)
        faces.append(face)
    return faces

def detectOneFace(gray_img, detector, flags=cv.CV_HAAR_SCALE_IMAGE, min_size=(64, 64)):
    rects = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3,
            minSize=min_size, flags=flags)
    if len(rects) == 0:
        return None
    
    sizes = [w * h for x, y, w, h in rects]
    largest_idx = np.argmax(sizes)
    x, y, w, h = rects[largest_idx]
    face = np.array(gray_img[y:y+h, x:x+w])
    face = cv2.resize(face, min_size, interpolation=cv.CV_INTER_AREA)
    return face

class FaceDetector(object):
    def __init__(self, detector, flags=cv.CV_HAAR_SCALE_IMAGE, min_size=(64, 64)):
        if isinstance(detector, str):
            self.detector_ = cv2.CascadeClassifier(detector)
        else:    
            self.detector_ = detector
        self.flags_ = flags
        self.min_size_ = min_size
        
    def detectFaces(self, gray_img):
        return detectFaces(gray_img, self.detector_, self.flags_, self.min_size_)

    def detectOneFace(self, gray_img):
        return detectOneFace(gray_img, self.detector_, self.flags_, self.min_size_)

if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    face = detectOneFace(toGray(img), sys.argv[1])
    cv2.imshow("test", face)
    cv2.waitKey()
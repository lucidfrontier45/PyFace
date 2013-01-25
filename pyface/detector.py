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

BIGGEST_FACE = cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv.CV_HAAR_DO_ROUGH_SEARCH
MANY_FACES = cv.CV_HAAR_SCALE_IMAGE

def detectFaces(gray_img, detector, flags=MANY_FACES, min_size=(64, 64)):
    rects = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3,
            minSize=min_size, flags=flags)
    faces = []
    for x, y, w, h in rects:
        face = np.array(gray_img[y:y+h, x:x+w])
        face = cv2.resize(face, min_size, interpolation=cv.CV_INTER_AREA)
        faces.append(face)
    return faces

def detectOneFace(gray_img, detector, flags=BIGGEST_FACE, 
                  min_size=(64, 64), resize="min_size"):
    
    roi = detectOneFaceROI(gray_img, detector, flags, min_size)
    if roi is None:
        return None
    x, y, w, h = roi
    face = np.array(gray_img[y:y+h, x:x+w])
    if resize =="min_size":
        face = cv2.resize(face, min_size, interpolation=cv.CV_INTER_AREA)
    elif isinstance(resize, (list, tuple)):
        face = cv2.resize(face, resize, interpolation=cv.CV_INTER_AREA)
    else:
        pass
    return face

def detectOneFaceROI(gray_img, detector, flags=BIGGEST_FACE,
                      min_size=(64, 64)):
    rects = detector.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3,
            minSize=min_size, flags=flags)
    if len(rects) == 0:
        return None
    
#    sizes = [w * h for x, y, w, h in rects]
#    largest_idx = np.argmax(sizes)
#    x, y, w, h = rects[largest_idx]
    
    x, y, w, h = rects[0]
    
    return (x, y, w, h)

class FaceDetector(object):
    def __init__(self, detector, min_size=(64, 64)):
        if isinstance(detector, str):
            self.detector_ = cv2.CascadeClassifier(detector)
        else:    
            self.detector_ = detector
        self.min_size_ = min_size
        
    def detectFaces(self, gray_img):
        return detectFaces(gray_img, self.detector_, MANY_FACES,
                            self.min_size_)

    def detectOneFace(self, gray_img, resize="min_size"):
        return detectOneFace(gray_img, self.detector_, BIGGEST_FACE,
                              self.min_size_, resize=resize)
        
    def detectOneFaceROI(self, gray_img):
        return detectOneFaceROI(gray_img, self.detector_, BIGGEST_FACE,
                            self.min_size_)

if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    face = detectOneFace(toGray(img), sys.argv[1])
    cv2.imshow("test", face)
    cv2.waitKey()
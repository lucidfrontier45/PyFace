'''
Created on 2012/12/07

@author: du
'''

import numpy as np
from pyface import recognizer, detector, utils

class FaceDetectRecognizer(object):
    def __init__(self, haar_detector_xml):
        self.detector_ = detector.FaceDetector(haar_detector_xml)
        self.recognizer_ = recognizer.LaplacianFace(5, 15)
        self.recognizer_.initByOlivetti()
        
    def detect(self, imgs, labels):
        faces = []
        face_labels = []
        for img, label in imgs, labels:
            face = self.detector_.detectOneFace(utils.toGray(img))
            if face:
                faces.append(face)
                face_labels.append(label)
        cvt_func = lambda x:utils.convImgMat(x, "numpy")
        faces = np.array(map(cvt_func, faces))
        
        return faces, face_labels
    
    def learn(self, imgs, labels):
        faces, face_labels = self.detect(imgs, labels)
        if len(faces) == 0:
            return 
        return self.recognizer_.fit_addtional(faces, face_labels)
    
    def predict(self, img):
        face = self.detector_.detectOneFace(utils.toGray(img))
        face = utils.convImgMat(face, "numpy")
        return self.recognizer_.predict(face)

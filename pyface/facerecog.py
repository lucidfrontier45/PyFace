'''
Created on 2012/12/07

@author: du
'''

import numpy as np
import cv, cv2
#from pyface import recognizer, detector, utils
from . import recognizer, detector, utils
try:
    import cPickle as pickle
except:
    import pickle


class FaceDetectRecognizer(object):
    def __init__(self, haar_detector_xml, m_recognizer=None):
        self._cascade_xml = haar_detector_xml
        self.detector_ = detector.FaceDetector(haar_detector_xml)
        if m_recognizer:
            self.recognizer_ = m_recognizer
        else:
            self.recognizer_ = recognizer.EigenFace(20)
            self.recognizer_.initByOlivetti()
        
    def detect(self, imgs, labels):
        faces = []
        face_labels = []
        for img, label in zip(imgs, labels):
            face = self.detector_.detectOneFace(utils.toGray(img))
            if not face is None:
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
        idx = self.recognizer_.retriveClosestIdx(face)[0]
        return idx, self.recognizer_._labels[idx]

    def save(self, fname):
        fp = open(fname, "wb")
        pickle.dump((self._cascade_xml, self.recognizer_), fp)
        fp.close()
        
    def getLargestLabel(self):
        return max(self.recognizer_._labels)

def learn(model, label, imgs):
    labels = [label] * len(imgs)
    model.learn(imgs, labels)

def predict(model, img):
    return model.predict(img)

if __name__ == "__main__":
    import sys
    if sys.argv[2] == "-i":
        model = FaceDetectRecognizer(sys.argv[3])
        model.save(sys.argv[1])
    else:
        cascade_xml, m_recognizer = pickle.load(open(sys.argv[1], "rb"))
        model = FaceDetectRecognizer(cascade_xml, m_recognizer)
        if sys.argv[2] == "-p":
            img = cv2.imread(sys.argv[3])
            idx, label = predict(model, img)
            print label
            face = model.recognizer_._faces[idx]
            face = utils.convImgMat(face, "opencv")
            cv2.imshow("input", img)
            cv2.imshow("closest.jpg", face)
            cv2.waitKey()
        elif sys.argv[2] == "-l":
            imgs = [cv2.imread(fname) for fname in sys.argv[4:]]
            learn(model, sys.argv[3], imgs)
            model.save(sys.argv[1])
        else:
            print model.getLargestLabel()

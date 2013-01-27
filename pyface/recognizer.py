'''
recognizer.py
Created on 2012/12/05
@author: du
'''

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.manifold.lpp import LocalityPreservingProjection as LPP
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets import fetch_olivetti_faces

class _BaseFaceRecognizer(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, transformer=None, classifier=None):
        if transformer:
            self.transformer_ = transformer
        else:
            self.transformer_ = PCA(n_components=0.8)
        
        if classifier:
            self.classifier_ = classifier
        else:
            self.classifier_ = KNeighborsClassifier(1)


    def transform(self, faces):
        return self.transformer_.transform(faces)

    def fit(self, faces, labels):
        self._faces = faces.copy()

        try:
            self.transformer_.fit(faces, labels)
        except TypeError:
            self.transformer_.fit(faces)
            
        features = self.transform(faces)
        self._features = features.copy()
        self._labels = list(labels)
        self.classifier_.fit(features, labels)
        return self
        
    def fit_addtional(self, faces, labels):
        if faces.shape[1] != self._faces.shape[1]:
            raise ValueError("additional data have wrong dimension, "+\
                             "expected %d but got %d\n"\
                             % (self._faces.shape[1], faces.shape[1]))
        features = self.transform(faces)
        self._faces = np.vstack((self._faces, faces))
        self._features = np.vstack((self._features, features))
        self._labels = self._labels + list(labels)
        self.classifier_.fit(self._features, self._labels)
        return self
        
           
    def predict(self, faces):
        features = self.transform(faces)
        return self.classifier_.predict(features)
    
    def retriveClosestIdx(self, faces):
        faces = np.asarray(faces)
        features = self.transform(faces)
        if isinstance(self.classifier_, KNeighborsClassifier):
            neigh_dist, neigh_ind = self.classifier_.kneighbors(features)
            return neigh_ind.flatten()
        else:
            return None
    
    def retriveClosestImage(self, faces):
        faces = np.asarray(faces)
        idx = self.retriveClosestIdx(faces)
        if idx:
            return self._faces[idx]
        else:
            return None
        
    def initByOlivetti(self):
        olivetti = fetch_olivetti_faces()
        faces = olivetti.data
        labels = [0] * len(faces)
        return self.fit(faces, labels)
    
class EigenFace(_BaseFaceRecognizer):
    def __init__(self, n_components=0.8, copy=True, whiten=False):
        transformer = PCA(n_components, copy, whiten)
        _BaseFaceRecognizer.__init__(self, transformer=transformer)

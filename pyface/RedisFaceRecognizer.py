'''
Created on 2012/12/09

@author: du
'''

import numpy as np
import cv, cv2
from scipy.spatial import distance
from sklearn.manifold.lpp import LPP
from sklearn.datasets import fetch_olivetti_faces
#from pyface import recognizer, detector, utils
from . import recognizer, detector, utils
try:
    import cPickle as pickle
except:
    import pickle
import redis

DUMMY_PATH = "dummy_path"

class RedisRecognizer(object):
    def __init__(self, haar_detector_xml, n_neighbors=5,
                  n_components=20, feature_coef=None,
                 host='localhost', port=6379, db=0, password=None,
                 socket_timeout=None, connection_pool=None,
                 charset='utf-8', errors='strict', unix_socket_path=None):
        self._cascade_xml = haar_detector_xml
        self.detector_ = detector.FaceDetector(haar_detector_xml)
        self.redis = redis.Redis(host, port, db, password, socket_timeout,
                        connection_pool, charset, errors, unix_socket_path)
        self.feature_coef_ = feature_coef
        
        self._n_neighbors = n_neighbors
        self._n_components = n_components

        init_flag = self.redis.get("face_init")
        if init_flag is None:
            self.init_face()

    def init_face(self):
        print "initialize redis face database"
        self.init_redis()
        self.init_features()
        self.redis.set("face_init", 1)

    def init_features(self):
        if self.feature_coef_ is None:
            self.feature_coef_ = self.redis.get("feature_coef")
        if self.feature_coef_ is None:
            lpp = LPP(self._n_neighbors, self._n_components)
            test_faces = fetch_olivetti_faces()
            features = lpp.fit_transform(test_faces.data)
            self.redis.set("name:0", "olivetti_faces")
            self.redis.set("name_id:olivetti_faces", 0)
            dim1, dim2 = lpp._components.shape
            self.redis.hmset("feature_coef", 
                    {"dim1":dim1, "dim2":dim2,
                     "data":lpp._components.tostring()})
            test_features = [f.tostring() for f in features]
            self.redis.rpush("features", *test_features)
            test_face_data = [np.array(f, dtype=np.float64).tostring() for f in test_faces.data]
            self.redis.rpush("faces", *test_face_data)
            for i in xrange(len(test_faces.data)):
                self.redis.hmset("picture:%d" % (i),
                                 {"name_id":0, "pic_path":DUMMY_PATH})
            self.redis.set("last_pic_id", len(test_faces.data) - 1)

    def init_redis(self):
        self.redis.delete("last_pic_id")
        self.redis.set("last_pic_id", 0)
        self.redis.delete("last_name_id")
        self.redis.set("last_name_id", 0)
        self.redis.delete("faces")
        self.redis.delete("features")
        self.redis.delete("feature_coef")
        keys = self.redis.keys("name_id:*")
        keys += self.redis.keys("name:*")
        keys += self.redis.keys("picture:")
        if len(keys) > 0:
            self.redis.delete(*keys)

    def detect(self, img, convert=True, resize="min_size"):
        face = self.detector_.detectOneFace(utils.toGray(img), resize=resize)
        if face is None:
            return None
        if convert:
            face = utils.convImgMat(face, "numpy")
        return face
    
    def detectROI(self, img):
        face_roi = self.detector_.detectOneFaceROI(utils.toGray(img))
        return face_roi

    def learn(self, img, img_path, name):
        face = self.detect(img)
        if face is None:
            return False

        feature_coef = self._getFeatureCoef()
        feature = np.dot(face, feature_coef)

        while 1:
            try:
                curr_last_name_id = int(self.redis.get("last_name_id"))
                curr_last_pic_id = int(self.redis.get("last_pic_id"))
                p = self.redis.pipeline()
                p.watch("name_id:%s" % (name))
                p.watch("last_name_id")
                p.multi()
                name_id = self.redis.get("name_id:%s" % (name))
                if name_id is None:
                    name_id = curr_last_name_id + 1
                    self.redis.set("last_name_id", name_id)
                    self.redis.set("name:%s" % name_id, name)
                    self.redis.set("name_id:%s" % name, name_id)
                p.execute()
                break
            except redis.WatchError:
                continue
        while 1:
            try:
                p.watch("last_pic_id")
                p.multi()
                p.rpush("features", feature.tostring())
                p.rpush("faces", face.tostring())
                pic_id = curr_last_pic_id + 1
                p.set("last_pic_id", pic_id)
                p.hmset("picture:%d"%pic_id, {"name_id":name_id, "pic_path":img_path})
                p.execute()
                break
            except redis.WatchError:
                continue
            
        print "hmset picture:%d {'name_id':%s, 'pic_path':%s}" \
                                                % (pic_id, name_id, img_path)
        return True

    def _getFeatures(self):
        features = self.redis.lrange("features", 0, -1)
        features = np.array([np.fromstring(f) for f in features])
        return features
    
    def _getFeatureCoef(self):
        feature_coef = self.redis.hgetall("feature_coef")
        dim1, dim2 = int(feature_coef["dim1"]), int(feature_coef["dim2"])
        feature_coef = np.fromstring(feature_coef["data"]).reshape(dim1, dim2)
        return feature_coef

    def predict(self, img):
        face = self.detector_.detectOneFace(utils.toGray(img))
        if face is None:
            return None
        face = utils.convImgMat(face, "numpy")
        features = self._getFeatures()
        feature_coef = self._getFeatureCoef()
        query_feature = np.dot(face, feature_coef)
        distances = distance.cdist(features, [query_feature]).flatten()
        closest_idx = distances.argmin()
        print "(idx, distance) = (%d, %f)" %(closest_idx, distances[closest_idx]) 
        print "hgetall picture:%d" %closest_idx
        closest_pic = self.redis.hgetall("picture:%d" %(closest_idx))
        print closest_pic
        name = self.redis.get("name:%s" % (closest_pic["name_id"]))
        closest_face = np.fromstring(model.redis.lindex("faces", closest_idx))
        closest_face = utils.convImgMat(closest_face, "opencv")
        closest_pic["name"] = name
        closest_pic["face"] = closest_face
        return closest_pic

if __name__ == "__main__":
    import sys
    import os.path
    import shutil
    HOME_DIR = "/home/du/.pyface/"
    model = RedisRecognizer("/home/du/workspace/OpenCV-2.4.2/data/haarcascades/haarcascade_frontalface_default.xml")
    if sys.argv[1] == "-i":
        model.init_face()
    if sys.argv[1] == "-p":
        img = cv2.imread(sys.argv[2])
        closest_pic = model.predict(img)
        if closest_pic is None:
            print "no face was found"
            sys.exit()
        if closest_pic["name_id"] == "0":
            print "not in database"
        else:
            cv2.imshow("input", img)
            closest_img = cv2.imread(closest_pic["pic_path"])
            x, y, w, h = model.detectROI(closest_img)
            closest_face = closest_img[y:y+h, x:x+w]
            cv2.imshow(closest_pic["name"], closest_face)
            cv2.waitKey()
    if sys.argv[1] == "-l":
        name = sys.argv[2]
        print "name = ", name
        for img_path in sys.argv[3:]:
            copy_path = os.path.join(HOME_DIR, os.path.basename(img_path))
            shutil.copyfile(img_path, copy_path)
            img = cv2.imread(copy_path)
            model.learn(img, copy_path, name)

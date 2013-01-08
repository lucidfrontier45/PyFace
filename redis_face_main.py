'''
Created on 2013/01/09

@author: du
'''

import sys
import os.path
import shutil
from pyface.RedisFaceRecognizer import RedisRecognizer
import cv2

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

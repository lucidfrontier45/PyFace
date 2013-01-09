'''
Created on 2013/01/09

@author: du
'''

import sys
import os.path
import shutil
from pyface.RedisFaceRecognizer import RedisRecognizer
import cv2
import pytc
import tempfile
import base64
import os.path

html_template = """<html>
<body>
<p>input</p>
<img src=\"data:image/jpg;base64,%s\">
<p>%s</p>
<img src=\"data:image/jpg;base64,%s\">
</body>
</html>"""


hdb = pytc.HDB()

def init(model):
    model.init_face()

def learn(model, name, img_paths, hdb_path):
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path)
            ret = model.learn(img, img_path, name)
            if ret:
                hdb.open(hdb_path, pytc.HDBOWRITER | pytc.HDBOCREAT)
                hdb.put(img_path, open(img_path, "rb").read())
                hdb.close()
#                copy_path = os.path.join(HOME_DIR, os.path.basename(img_path))
#                shutil.copyfile(img_path, copy_path)
        except:
            pass
        
def predict(model, img_path, hdb_path):
    img = cv2.imread(img_path)
    closest_pic = model.predict(img)
    if closest_pic is None:
        print "no face was found"
        sys.exit()
    if closest_pic["name_id"] == "0":
        print "not in database"
    else:
#        cv2.imshow("input", img)
        name = closest_pic["name"]
        hdb.open(hdb_path, pytc.HDBOREADER)
        input_dat = base64.b64encode(open(img_path, "rb").read())
        closest_pic_dat = hdb.get(closest_pic["pic_path"])
        
        tfp = tempfile.NamedTemporaryFile(prefix=os.path.splitext(
                        closest_pic["pic_path"])[-1])
        tfp.file.write(closest_pic_dat)
        closest_img = cv2.imread(tfp.name)
        x, y, w, h = model.detectROI(closest_img)
        closest_face = closest_img[y:y+h, x:x+w]
        ret, buf = cv2.imencode(".JPG", closest_face)
#        cv2.imshow(closest_pic["name"], closest_face)
#        cv2.waitKey()
        html = html_template % (input_dat, name, base64.b64encode(buf))
        fp = open("out.html", "wb")
        fp.write(html)
        fp.close()
        print name

HOME_DIR = "/home/du/.pyface/"
model = RedisRecognizer("/home/du/workspace/OpenCV-2.4.2/data/haarcascades/haarcascade_frontalface_default.xml")
hdb_path = "/home/du/.pyface/face.hdb"
if sys.argv[1] == "-i":
    init(model)
if sys.argv[1] == "-p":
    predict(model, sys.argv[2], hdb_path)
if sys.argv[1] == "-l":
    name = sys.argv[2]
    print "name = ", name
    learn(model, name, sys.argv[3:], hdb_path)

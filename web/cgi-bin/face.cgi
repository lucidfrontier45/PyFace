#!/usr/bin/python

'''
Created on 2013/01/09

@author: du
'''

import cgi
import cgitb
import sys
import os, os.path
import shutil
from pyface.RedisFaceRecognizer import RedisRecognizer
import cv2
import pytc
import tempfile
import base64
import os.path
import time
import json

html_template = """<p>input</p>
<img src=\"data:image/jpg;base64,%s\">
<p>%s</p>
<img src=\"data:image/jpg;base64,%s\">
"""

html_template2 = """<p>input</p>
<img src=\"data:image/jpg;base64,%s\">
<p>%s</p>"""


hdb = pytc.HDB()

def init(model):
    model.init_face()

def learn(model, name, img_paths, hdb_path):
    msg = [] 
    for img_path in img_paths:
        try:
            img = cv2.imread(img_path)
            ret = model.learn(img, img_path, name)
            if ret:
                hdb.open(hdb_path, pytc.HDBOWRITER | pytc.HDBOCREAT)
                hdb.put(img_path, open(img_path, "rb").read())
                hdb.close()
                msg.append("%s learned" % img_path)
#                copy_path = os.path.join(HOME_DIR, os.path.basename(img_path))
#                shutil.copyfile(img_path, copy_path)
            else :
                msg.append("%s not used" % img_path)
        except:
            pass
    return json.dumps({"result":200, "msg":msg})
        
def predict(model, img_path, hdb_path):
    img = cv2.imread(img_path)
    input_dat = base64.b64encode(open(img_path, "rb").read())
    closest_pic = model.predict(img)
    if closest_pic is None:
        result_html = html_template2 % (input_dat, "no face was found")
        return json.dumps({"result":204, "msg":result_html})
    elif closest_pic["name_id"] == "0":
        result_html = html_template2 % (input_dat, "not in database ")
        return json.dumps({"result":204, "msg":result_html})
    else:
#        cv2.imshow("input", img)
        name = closest_pic["name"]
        hdb.open(hdb_path, pytc.HDBOREADER)
        closest_pic_dat = hdb.get(closest_pic["pic_path"])
        hdb.close()
        
        tfp = tempfile.NamedTemporaryFile(prefix=os.path.splitext(
                        closest_pic["pic_path"])[-1])
        tfp.file.write(closest_pic_dat)
        closest_img = cv2.imread(tfp.name)
        x, y, w, h = model.detectROI(closest_img)
        closest_face = closest_img[y:y+h, x:x+w]
        ret, buf = cv2.imencode(".JPG", closest_face)
#        cv2.imshow(closest_pic["name"], closest_face)
#        cv2.waitKey()
        result_html = html_template % (input_dat, name, base64.b64encode(buf))
        return json.dumps({"result":200, "msg":result_html}) 
    #return json.dumps({"result":200, "input_data":input_data, 
    #    "name":name, "closest":base64.b64encode(buf)})

_buf_size = 1024 * 1024

def savePostedImage(file_form):
    ext = os.path.splitext(file_form.filename)[-1]
    tfp = tempfile.NamedTemporaryFile(suffix=ext)
    while True:
        chunk = file_form.file.read(_buf_size)
        if not chunk:
            break
        tfp.file.write(chunk)
    tfp.file.flush()
    return tfp

def getFiles(form):
    files = form["file[]"]
    if isinstance(files, list):
        filenames = ",".join([fp.filename for fp in files])
        tempfiles = [savePostedImage(fp) for fp in files]
    else:
        #filenames = files.filename.decode("utf-8")
        filenames = files.filename
        tempfiles = [savePostedImage(files)]
    return tempfiles

##########################################
#main
##########################################

cgitb.enable()

print "Content-Type: text/json; charset=UTF-8\r\n"

form = cgi.FieldStorage()
mode = form["mode"].value

HOME_DIR = "/home/pyface/"
hdb_path = "/home/pyface/face.hdb"

haar_cascade_dir = "/usr/local/share/OpenCV/haarcascades/"
model = RedisRecognizer(os.path.join(haar_cascade_dir, 
    "haarcascade_frontalface_default.xml"))

if mode == "init":
    init(model)
    try:
        os.remove(hdb_path)
    except:
        pass
    result = json.dumps({"result":200, "msg":"initialized"})
    print result
if mode == "predict":
    tempfiles = getFiles(form)
    result = predict(model, tempfiles[0].name, hdb_path)
    print result
if mode == "learn":
    tempfiles = getFiles(form)
    name = form["name"].value.decode("utf-8")
    result = learn(model, name, [tf.name for tf in tempfiles] , hdb_path)
    print result
    

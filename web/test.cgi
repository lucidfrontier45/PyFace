#!/usr/bin/python

import cgi
import cgitb
import tempfile
import os.path
import time

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

cgitb.enable()

print "Content-Type: text/plain; charset=UTF-8\r\n"

form = cgi.FieldStorage()
name = form["name"].value.decode("utf-8")
files = form["file[]"]
if isinstance(files, list):
    filenames = ",".join([fp.filename for fp in files])
    tempfiles = [savePostedImage(fp) for fp in files]
else:
    #filenames = files.filename.decode("utf-8")
    filenames = files.filename
    tempfiles = [savePostedImage(files)]



print "obtained name = " + name.encode("utf-8")
#print "obtained filenames = " + filenames.encode("utf-8")
print "obtained filenames = " + filenames

time.sleep(30)

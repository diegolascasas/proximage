#!/usr/bin/env python2
import cv2
import urllib2
import numpy as np
import sys
import contextlib

def _str2img(string):
    array  = np.fromstring(string, dtype=np.uint8)
    return cv2.imdecode(array,cv2.CV_LOAD_IMAGE_COLOR)

# from http://stackoverflow.com/questions/21702477/how-to-parse-mjpeg-http-stream-from-ip-camera
class WifiCapturer:
    def __init__(self, host, photos = True):
        if photos:
            self.hoststr = 'http://' + host + '/shot.jpg'
            self.read = self._read_photo
            self.source=urllib2.urlopen(self.hoststr)
        else:
            self.hoststr = 'http://' + host + '/video'
            self.read = self._read_video
        self.buf = '' # buffer to store the read bytes
            
    def _read_video(self):
        "parses the video stream"
        ## TODO: implement when no value is returned
        jpgstr = ''
        sucess = True
        while not jpgstr:
            newdata = self.source.read(1024)
            self.buf += newdata
            a = self.buf.find('\xff\xd8')
            b = self.buf.find('\xff\xd9')
            if a!=-1 and b!=-1:
                jpgstr   = self.buf[a:b+2]
                self.buf = self.buf[b+2:]
                success = True
        return sucess, _str2img(jpgstr)

    def _read_photo(self):
        "uses snapshots instead of video stream. slower, better res"
        with contextlib.closing(urllib2.urlopen(self.hoststr)) as source:
            jpgstr = source.read()
        return True, _str2img(jpgstr)
    
    
    def stream(self):
        while True:
            i = self.read()
            if not i:
                break
            cv2.imshow(self.hoststr,i)
            if cv2.waitKey(1) == 27:
                break

    def release(self):
        self.buf = ""
        self.source.close()

if __name__ == '__main__':
    if len(sys.argv)>1:
        host = sys.argv[1]
    else: 
        host = "192.168.25.3:8080"

    cap = WifiCapturer(host)
    ## print 'Streaming ' + cap.hoststr
    cap.stream()

#!/usr/bin/env python
import sys
import numpy as np
import cv2
from ipwebcam  import WifiCapturer

DATADIR = "/usr/local/opt/opencv/share/OpenCV/haarcascades/"
MODELS = [
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
    "haarcascade_frontalface_default.xml",
]

# IMAGE_SCALE = 1.2
HAAR_SCALE = 1.2  # 1.2
MIN_NEIGHBORS = 4 # 2
HAAR_FLAGS = 0 # cv2.CASCADE_DO_CANNY_PRUNING
EPOCH_SIZE = 30
# - reescalonar
# - 13 x 10

def rect_overlap(a, b):
    ax,ay,aw,ah = a
    bx,by,bw,bh = b

    dx = min(ax+aw, bx+bw) - max(ax,bx)
    dy = min(ay+ah, by+bh) - max(ay,by)

    if (dx < 0) and (dy < 0):
        return 0
    i = dx*dy
    overlap = i / float(aw*ah + bw*bh - i)
    print "Overlap:",overlap
    return overlap

    
class FaceExtractor(object):
    def __init__(self, model_files, exclusion_overlap = 1):
        self.face_detectors = [cv2.CascadeClassifier(model_file)
                               for model_file in model_files]
        self.epoch  = 0
        self.nfaces = 0
        self.exclusion_overlap = exclusion_overlap
        self.faces_known = []
        self.forgetting_time = 3

    def naming_policy(self):
        path = "faces/%.3d.jpg" % (self.nfaces % 1000)
        self.nfaces += 1
        return path
        
    def _store_face(self, img, face):
        x,y,w,h = face
        region = img[y:y+h,x:x+w]
        path   = self.naming_policy()
        cv2.imwrite(path,region,[cv2.IMWRITE_JPEG_QUALITY,100])
        self.last_seen_face = face
        print "Written", path 

        
    def _must_be_stored(self, face):
        self.faces_known = filter(
            lambda x: self.epoch - x[1] < self.forgetting_time,
            self.faces_known
        )
        is_new = all(
            rect_overlap(face, other_face) < self.exclusion_overlap
            for other_face, epoch in self.faces_known
        )
        if is_new:
            self.faces_known.append((face,self.epoch))
            return True
        return False

    
    def scan(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_lists = [d.detectMultiScale(gray, 2, 4)
                      for d in self.face_detectors]
        for faces in face_lists:
            if len(faces) > 0:
                for face in faces:
                    if self._must_be_stored(face):
                        self._store_face(img, face)

        self.epoch +=1
        return faces


    # TODO: similares
class App(object):
    def __init__(self, source, model, epoch_size):
        if type(source) == type(0):
            self.cap = cv2.VideoCapture(source)
        elif type(source) == type(""):
            if "." in source and ":" in source:
                self.cap = WifiCapturer(source)
        self.d   = FaceExtractor(model, exclusion_overlap = 0.8)
        self.epoch_size = epoch_size

    def run(self, source = 0):
        i = 0
        try:
            while True:
                # Capture frame-by-frame
                ret, img = self.cap.read()
                if not ret:
                    break
                
                if i % self.epoch_size == 0:
                    faces = self.d.scan(img)
                i += 1

                for x,y,w,h in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

                # Display the resulting frame
                cv2.imshow('frame',img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # When everything done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(sys.argv)>1:
        source = sys.argv[1]
    else:
        source = 0
    
    models = (DATADIR + model for model in MODELS)
    App(source, models, EPOCH_SIZE).run()

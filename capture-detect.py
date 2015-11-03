#!/usr/bin/env python
import numpy as np
import cv2
from multiprocessing import Pool
from functools import partial

DATADIR = "/usr/local/opt/opencv/share/OpenCV/haarcascades/"
MODELS = [
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
    "haarcascade_frontalface_default.xml",
]

# IMAGE_SCALE = 1.2
HAAR_SCALE = 2  # 1.2
MIN_NEIGHBORS = 5 # 2
HAAR_FLAGS = 0 # cv2.CASCADE_DO_CANNY_PRUNING
EPOCH_SIZE = 30

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
        self.face_detectors = [cv2.CascadeClassifier(model_file) for model_file in model_files]
        self.epoch  = 0
        self.nfaces = 0
        self.exclusion_overlap = exclusion_overlap
        self.faces_known = []
        self.forgetting_time = 3
        self.workers = Pool(len(self.face_detectors))

        
    def _store_face(self, img, face):
        self.nfaces += 1
        x,y,w,h = face
        region = img[y:y+h,x:x+w]
        path   = "faces/%.4d.jpg" % self.nfaces
        cv2.imwrite(path,region)
        self.last_seen_face = face
        print "Writen", path 

        
    def _must_be_stored(self, face):
        self.faces_known = filter(lambda x: self.epoch - x[1] < self.forgetting_time, self.faces_known)
        is_new = all(rect_overlap(face, other_face) < self.exclusion_overlap
                     for other_face, epoch in self.faces_known)
        if is_new:
            self.faces_known.append((face,self.epoch))
            return True
        return False

    
    def scan(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        face_lists = [d.detectMultiScale(gray, 2, 4) for d in self.face_detectors]
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
        self.cap = cv2.VideoCapture(source)
        self.d   = FaceExtractor(model, exclusion_overlap = 0.8)
        self.epoch_size = epoch_size
        
    def run(self, source = 0):
        i = 0
        try:
            while True:
                # Capture frame-by-frame
                ret, img = self.cap.read()

                
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
    models = (DATADIR + model for model in MODELS)
    App(0, models, EPOCH_SIZE).run()

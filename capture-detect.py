#!/usr/bin/env python
import sys
import numpy as np
import cv2
from ipwebcam  import WifiCapturer
from argparse import ArgumentParser

DATADIR = "/usr/local/opt/opencv/share/OpenCV/haarcascades/"

EXCLUSION_OVERLAP = 0.7
# IMAGE_SCALE = 1.2
HAAR_SCALE = 1.2  # 1.2
MIN_NEIGHBORS = 4 # 2
HAAR_FLAGS = 0 # cv2.CASCADE_DO_CANNY_PRUNING
EPOCH_SIZE = 30

## NOTAS:
## - Ainda nao testei o body_extractor, 

def rect_overlap(a, b):
    ax,ay,aw,ah = a
    bx,by,bw,bh = b

    dx = min(ax+aw, bx+bw) - max(ax,bx)
    dy = min(ay+ah, by+bh) - max(ay,by)

    if (dx < 0) or (dy < 0):
        print "Overlap: 0"
        return 0
    i = dx*dy
    overlap = i / float(aw*ah + bw*bh - i)
    print "Overlap:",overlap
    return overlap

    
class EntityExtractor(object):
    def __init__(self, name, model_files, exclusion_overlap = 1, forget = 3):

        self.entity_detectors = [cv2.CascadeClassifier(model_file)
                               for model_file in model_files]

        self.bg = cv2.BackgroundSubtractorMOG2()

        self.exclusion_overlap = exclusion_overlap
        self.entity_name       = name
        self.forgetting_time   = forget

        self.entities_known = []
        self.epoch     = 0
        self.nentities = 0


    def naming_policy(self):
        path = "entities/%s-%d.jpg" % (self.entity_name, self.nentities)
        self.nentities += 1
        return path
        
    def _store_entity(self, entity):
        x,y,w,h = entity
        path   = self.naming_policy()

        region = self.img[y:y+h,x:x+w]
        region_mask = self.mask[y:y+h,x:x+w]
        # print region.shape, region_mask.shape
        # region &= region_mask.reshape(h,w,1)
        
        cv2.imwrite(path,region,[cv2.IMWRITE_JPEG_QUALITY,100])
        cv2.imwrite(path + '.mask.jpg',region_mask)
        print "Written", path 

        
    def _must_be_stored(self, entity):
        self.entities_known = filter(
            lambda x: self.epoch - x[1] < self.forgetting_time,
            self.entities_known
        )
        is_new = all(
            rect_overlap(entity, other_entity) < self.exclusion_overlap
            for other_entity, epoch in self.entities_known
        )
        if is_new:
            self.entities_known.append((entity,self.epoch))
            return True
        return False

    
    def scan(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        entity_lists = [list(d.detectMultiScale(gray, 2, 4))
                      for d in self.entity_detectors]

        entities = sum(entity_lists,[])

        if len(entities) > 0:
            self.img  = img
            self.mask = self.bg.apply(img)
            for entity in entities:
                if self._must_be_stored(entity):
                    self._store_entity(entity)

        self.epoch +=1
        return entities



class App(object):
    def __init__(self, source, detectors, epoch_size, display = False):

        if type(source) == type(0):
            self.cap = cv2.VideoCapture(source)
        elif type(source) == type(""):
            if "." in source and ":" in source:
                self.cap = WifiCapturer(source)

        self.detectors = detectors
 
        self.epoch_size = epoch_size
        
        self.colors = [(255,0,0), (0,255,0)] # gambiarra, consertar depois

        self.display = display

        
    def run(self, source = 0):
        i = 0
        try:
            while True:
                # Capture frame-by-frame
                ret, img = self.cap.read()
                if not ret:
                    break

            
                if i % self.epoch_size == 0:
                    entities = [d.scan(img) for d in self.detectors]
                
                if self.display:
                    for ent_type, color in zip(entities, self.colors):
                        for x,y,w,h in ent_type:
                            cv2.rectangle(img, (x,y), (x+w,y+h), color,2)
                    cv2.imshow('frame',img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                sys.stdout.flush() ## avoid holding the print 
                i += 1
        finally:
            # When everything done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-s','--source',default=0,
                        help='IP of the source if using ipcamera')
    parser.add_argument('-d','--display',action='store_true',
                        help='Whether to display the captured input')

    args = parser.parse_args()

    face_models = (DATADIR + model for model in
                   ["haarcascade_profileface.xml",
                    "haarcascade_frontalface_default.xml"])
    face_extractor = EntityExtractor("face", face_models,
                                     exclusion_overlap = EXCLUSION_OVERLAP)

    body_models = (DATADIR + model for model in
                   ["haarcascade_fullbody.xml"])
    body_extractor = EntityExtractor("body", body_models,
                                     exclusion_overlap = EXCLUSION_OVERLAP)
    
    App(args.source, [face_extractor, body_extractor],
        EPOCH_SIZE, args.display).run()

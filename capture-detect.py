#!/usr/bin/env python

## Pacotes para lidar com o sistema operacional
import sys
import time
import os
from argparse import ArgumentParser


## Pacotes para processamento de imagem
import numpy as np
import cv2
import dlib
from skimage.draw import polygon

## Outros
from ipwebcam  import WifiCapturer
import OSC


######### VARIAVEIS GLOBAIS PARA CONFIGURACAO DO SCRIPT  #########

## Tamanho da imagem ---------------------------------------------
## Largura: RESIZE[0] + PADSIZE[0 e 2],
## Altura: RESIZE[1] + PADSIZE[1 e 3]
### Resize diz o tamanho que a imagem original tera
RESIZE  = (500,500)
### Padsize diz sobre o tamanho das margens Esq., Cima, Dir., Baixo
PAD_SIZE = (150,150,150,150)


## Tamanho minimo aceitavel do rosto -----------------------------
MIN_SIZE_FACE = (250, 250)


## Espera minima para capturar cada rosto ------------------------
## Se um rosto for capturado a menos tempo que SLEEP_TIME de outro,
##  ele eh jogado fora,
SLEEP_TIME = 5


## Mensagens OSC -------------------------------------------------
### Espera em segundos para mandar cada mensagem
MESSAGE_DELAY = 1
### Numero de imagens
HOLDBACK = 10


## Esse eh o diretorio do sistema que prediz a forma do rosto.
## Nao mudar a menos que as coisas mudem.
PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat"


##################################################################

class OSCMessager(object):
    """ Manages osc messages to a specific endpoint   """

    def __init__(self, endpoint, host='127.0.0.1', port=8001):
        self.client = OSC.OSCClient()
        self.client.connect((host, port))
        self.endpoint = endpoint

    def send(self, msg):
        oscmsg = OSC.OSCMessage()
        oscmsg.setAddress(self.endpoint)
        oscmsg.append(msg)
        self.client.send(oscmsg)


class HoldbackMessager(OSCMessager):
    """ Waits for a couple of mesages before sending them """

    def __init__(self,
                 holdback, message_delay,
                 endpoint, host='127.0.0.1', port=8001):
        self.holdback = holdback
        self.message_delay = message_delay
        self.held_msgs = []
        super(HoldbackMessager, self)(endpoint, host, port)

    def send(self, msg):
        print "Written", msg
        self.held_msgs.append(msg)
        if len(held_msgs) >= self.holdback:
            for msg in self.held_msgs:
                super(HoldbackMessager, self).send(msg)
                time.sleep(self.message_delay)
            self.held_msgs = []


class StorageHandler(object):
    def __init__(self, prefix,
                 dest_dir = 'entities',
                 erase_bg = True,
                 sleep = 10,
                 min_size = (100, 100),
                 post_process = None,
                 messager = None,
                 extension = 'png'):

        self.dest_dir = dest_dir
        self.prefix = prefix
        self.erase_bg = erase_bg
        self.sleep = sleep
        self.min_size = min_size
        self.extension = extension

        self.messager = None

        if post_process is None:
            self.post_process = lambda x: x
        else:
            self.post_process = post_process

        if extension == 'jpg':
            ## 0 to 100, Higher means better, default 95
            self.imwrite_params = [cv2.IMWRITE_JPEG_QUALITY,100]
        elif extension == 'png':
            ## 0 to 9, smaller means less compression time, default 3
            self.imwrite_params = [cv2.IMWRITE_PNG_COMPRESSION,3]
        else:
            raise ValueError('Unsupported Extension')


        self.nentities = len(os.listdir(dest_dir))
        self.timer = time.time()


    def naming_policy(self):
        """ How the extracted entities must be named in the filesystem """
        path = "%s/%s%d.%s" % (
            self.dest_dir,
            self.prefix,
            self.nentities,
            self.extension
        )
        self.nentities += 1
        return path


    def qualified_for_storage(self, entity):
        """ Checks if entity meets criteria for storage """
        t = time.time()
        if (t - self.timer < self.sleep or
            self.min_size[0] > entity.shape[0] or
            self.min_size[1] > entity.shape[1] ):
            return False

        self.timer = t
        return True


    def store_entity(self, entity):
        """ Stores the region of an entity."""
        path   = self.naming_policy()
        cv2.imwrite(path,self.post_process(entity),self.imwrite_params)
        if self.messager is not None:
            messager.send(path)


    def handle(self, entity):
        ent = entity.crop(erase_bg=self.erase_bg)
        if self.qualified_for_storage(ent):
            self.store_entity(ent)
        # self.epoch +=1


class FaceLandmark(object):
    def __init__(self, pts, ref_im):
        self.im = ref_im
        max_y, max_x  = ref_im.shape[:2]
        self.data = np.matrix([
            [min(p.x,max_x), min(p.y,max_y)]
            for p in pts.parts()
        ])

    def draw(self, im, color = (255, 255,0), pt_size = 2):
        for p in self.data:
            pos = (p[0,0], p[0,1])
            cv2.circle(im, pos, pt_size, color=color)


    def get_contour(self):
        hullIndex = cv2.convexHull(self.data, returnPoints = False)
        hull =  self.data[hullIndex[:,0]]
        return polygon(hull[:,0],hull[:,1])


    def crop(self, erase_bg = False):
        cc, rr = self.get_contour()
        new_im = self.im[min(rr):max(rr), min(cc):max(cc), :].copy()
        if erase_bg:
            new_im = np.zeros(new_im.shape,dtype=np.uint8)
            new_im[rr - min(rr) - 1,cc - min(cc) - 1,:] = self.im[rr,cc,:]
        return new_im



class FaceDetectorWrapper(object):

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)


    def detect(self, im, translate = True):
        ent = self.detector(im, 1)
        if not translate:
            return ent
        if ent is None:
            return []
        else:
            return [(r.left(), r.top(), r.width(), r.height())
                    for r in ent]


    def predict(self, im, region):
        return FaceLandmark(self.predictor(im,region), im)


    def scan(self, im):
        faces = self.detect(im, translate=False)
        return [self.predict(im, face) for face in faces]



class App(object):
    """
    Facade object to manage detectors, image capture devices and
    OpenCV display's manager.
    Source can be either a number or an IP adress where IPWebcam is connected

    """

    def __init__(self, source,
                 epoch_size = 10, detectors = None,
                 storage_handler = None, display = False):


        if "." in source and ":" in source:
            self.cap = WifiCapturer(source)
        else:
            self.cap = cv2.VideoCapture(int(source))

        self.detectors = [] or detectors
        self.storage_handler = storage_handler
        self.epoch_size = epoch_size
        self.display = display


    def run(self, source = 0):
        # TODO: run asyncronously
        i = 0
        entities = []
        try:
            while True:
                # Capture frame-by-frame
                ret, im = self.cap.read()
                if not ret:
                    break

                if i % self.epoch_size == 0:
                    if self.detectors:
                        entities = sum(
                            [d.scan(im) for d in self.detectors], []
                        )

                for ent in entities:

                    if self.storage_handler is not None:
                        self.storage_handler.handle(ent)

                    if self.display:
                        ent.draw(im)


                cv2.imshow('frame',im)
                entities = []
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                sys.stdout.flush() ## avoid holding the print
                i += 1
        finally:
            # When everything done, release the capture
            self.cap.release()
            cv2.destroyAllWindows()


def resize_and_pad(im):
    # TODO: checar se a ordem eh essa mesma
    im = cv2.resize(im, RESIZE)

    pad_left, pad_top, pad_right, pad_bot = PAD_SIZE
    h,w = RESIZE
    newsize = (h + pad_top + pad_bot,
               w + pad_left + pad_right,
               im.shape[2])

    new_im = np.zeros(newsize, dtype=im.dtype)
    new_im[pad_top:pad_top+h, pad_left:pad_left+w, :] = im
    return new_im


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-s','--source',default='0',
                        help='IP of the source if using ipcamera')
    parser.add_argument('-d','--display',action='store_true',
                        help='Whether to display the captured input')

    args = parser.parse_args()

    msg = HoldbackMessager(
        endpoint = '/banana',
        message_delay = MESSAGE_DELAY,
        holdback = HOLDBACK
    )

    # TODO: Make storing and message passing more orthogonal
    storage = StorageHandler(
        prefix = '',
        post_process = resize_and_pad,
        min_size = MIN_SIZE_FACE,
        sleep = SLEEP_TIME,
        messager = msg
    )

    detectors = [FaceDetectorWrapper()]

    app = App(
        args.source, 10, detectors, storage, args.display
    )
    app.run()

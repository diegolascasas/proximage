#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np
import random
import glob
import OSC
from time import time



##### PARAMETROS DAS IMAGENS PRINCIPAIS #####

SPECIAL_FACE_DIR = "special"   ## pasta onde as imagens vão ser salvas
SPECIAL_FACE_BUFFER_SIZE = 13  ## numero de imagens principais antes de comecar a jogar as imagens para o background

SPECIAL_WIDTH  = 800  ## Largura das imagens principais
SPECIAL_HEIGHT = 800  ## Altura das imgens principais



##### PARAMETROS DO BACKGROUND #####

BG_WIDTH  =  3780       ## Largura do background
BG_HEIGHT =  1080       ## Altura do background
BG_FILE = "special/background-%d.jpg" ## Arquivo onde o background será salvo. %d permite que mais de um arquivo de background exista.

BG_IMAGE_SCALE = 0.5    ## Escala em que as imagens coladas no fundo vao ser redimensionadas
BG_TIME_THRESHOLD = 60  ## Tempo de espera antes de voltar a pegar imagens principais



def set_segment(img_size, max_value):
    offset = random.randrange(max_value)
    limit  = offset + img_size
    crop   = img_size
    if limit > max_value:
        limit = max_value 
        crop  = limit - offset
    return offset, limit, crop

def write(image, path):
    cv2.imwrite(path,image)

def send_message(msg):
    oscmsg = OSC.OSCMessage()
    oscmsg.setAddress("/detector")
    oscmsg.append(msg)
    osc.send(oscmsg)

## use current bg-image or create a new one.
bg_images = 0
bg_image = cv2.imread(BG_FILE % 0)
if bg_image is None:
    bg_image = np.zeros((BG_HEIGHT,BG_WIDTH,3), np.uint8)

## check the current number of "special images" in the directory
special_images = len(glob.glob('special/*.jpg'))

osc = OSC.OSCClient()
osc.connect(('127.0.0.1', 8001))

special_buffer = 0
last_time_checked = time()

for line in iter(sys.stdin.readline, ''):
    ## Parse output, look for selected lines
    line = line.strip()
    if "dropped frame" in line:
        continue

    print line
    if "Written" not in line:
        continue
    __, path = line.split()

    ## Read image
    img = cv2.imread(path)
    if img is None:
        print "could not read image", path
        continueqqq
    
    if special_buffer < SPECIAL_FACE_BUFFER_SIZE:
        ## Write image to directory
        newpath = "special/%d.jpg" % (special_images + 1)
        newsize = (SPECIAL_HEIGHT,SPECIAL_WIDTH)
        img = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)
        write(img,newpath)
        print "sent {} to {}".format(path, newpath)
        ## update state
        special_images += 1
        special_buffer += 1
        last_time_checked = time()
        ## Tell app if 13 new images were obtained
        if special_images % 13 == 0:
            send_message(1)
    else:
        ## apply mask
        h, w, __ = img.shape
        mask = cv2.imread(path + ".mask.jpg")
        __, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img &= mask

        ## resize image
        h = int(h * BG_IMAGE_SCALE)
        w = int(w * BG_IMAGE_SCALE)
        img = cv2.resize(img, (h,w), interpolation = cv2.INTER_AREA)

        ## find a random place to put the image in the background
        h_offset, h_limit, h_crop = set_segment(h, BG_HEIGHT)
        w_offset, w_limit, w_crop = set_segment(w, BG_WIDTH)
        crop = img[:h_crop, :w_crop, :] ## crop image if borders are outside background

        ## insert image in selected place
        bg_image[h_offset:h_limit, w_offset:w_limit,:] = crop

        ## save image
        write(bg_image,BG_FILE % (bg_images % 2))
        send_message(2)
        print "sent {} to background".format(path)

        ## update state
        bg_images += 1
        ## Check whether to go back recording special images
        if (time() - last_time_checked) > BG_TIME_THRESHOLD:
            special_buffer  = 0

#!/usr/bin/env python
import cv2
import sys
import numpy as np
import random
import glob

SPECIAL_FACE_BUFFER_SIZE = 2
SPECIAL_FACE_DIR = "special"

BG_FACE_BUFFER_SIZE = 2
BG_SCALE  = 100
BG_WIDTH  = 20 * BG_SCALE
BG_HEIGHT =  7 * BG_SCALE
BACKGROUND_FILE = "background.jpg"

def set_segment(img_size, max_value):
    offset = random.randrange(max_value)
    limit  = offset + img_size
    crop = img_size
    if limit > max_value:
        limit = max_value #(limit - max_value)
        crop  = limit - offset
    return offset, limit, crop

def write(image, path):
    cv2.imwrite(path,image)



special_buffer = 0
bg_buffer = 0

## use current bg-image or create a new one.
bg_image = cv2.imread(BACKGROUND_FILE)
if bg_image is None:
    bg_image = np.zeros((BG_HEIGHT,BG_WIDTH,3), np.uint8)

## check the current number of "special images" in the directory
special_images = len(glob.glob('special/*.jpg'))

while True:
    line  = sys.stdin.readline()

    ## Parse output, look for selected lines
    line = line.strip()
    print line
    if "Written" not in line:
        continue
    __, path = line.split()

    ## Read image
    img = cv2.imread(path)
    if img is None:
        print "could not read image", path
        continue
    
    if special_buffer < SPECIAL_FACE_BUFFER_SIZE:
        ## Write image to directory
        newpath = "special/%d.jpg" % (special_images + 1)
        write(img,newpath)
        print "sent {} to {}".format(path, newpath)

        ## update state
        special_images +=1
        special_buffer +=1

    else:
        ## apply mask
        h, w, __ = img.shape
        mask = cv2.imread(path + ".mask.jpg")
        __, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img &= mask

        ## find a random place to put the image in the background
        h_offset, h_limit, h_crop = set_segment(h, BG_HEIGHT)
        w_offset, w_limit, w_crop = set_segment(w, BG_WIDTH)
        crop = img[:h_crop, :w_crop, :] ## crop image if borders are outside background

        ## insert image in selected place
        bg_image[h_offset:h_limit, w_offset:w_limit,:] = crop

        ## save image
        write(bg_image,BACKGROUND_FILE)
        print "sent {} to background".format(path)

        ## update state
        bg_buffer += 1
        if bg_buffer > BG_FACE_BUFFER_SIZE:
            bg_buffer = 0
            special_buffer  = 0

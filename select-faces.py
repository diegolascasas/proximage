#!/usr/bin/env python
import cv2
import sys
import numpy as np
import random

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


special_buffer = special_faces = 0
bg_buffer = 0
bg_image = cv2.imread(BACKGROUND_FILE)
if bg_image is None:
    bg_image = np.zeros((BG_HEIGHT,BG_WIDTH,3), np.uint8)
for line in sys.stdin:
    if "Written" not in line:
        continue
    __, path = line.strip().split()
    img = cv2.imread(path)
    if img is None:
        print "could not read image", path
        continue
    if special_faces < SPECIAL_FACE_BUFFER_SIZE:
        newpath = "special/%d.jpg" % special_faces
        write(img,newpath)
        special_faces +=1
        special_buffer +=1
        print "sent {} to {}".format(path, newpath)
    else:
        w_offset, w_limit, w_crop = set_segment(img.shape[1], BG_WIDTH)
        h_offset, h_limit, h_crop = set_segment(img.shape[0], BG_HEIGHT)
        crop = img[:h_crop, :w_crop, :]
        
        bg_image[h_offset:h_limit, w_offset:w_limit,:] = crop

        write(bg_image,BACKGROUND_FILE)
        print "sent {} to background".format(path)
        bg_buffer += 1
        if bg_buffer > BG_FACE_BUFFER_SIZE:
            bg_buffer = 0
            special_buffer  = 0

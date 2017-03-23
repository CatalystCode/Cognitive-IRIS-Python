#!/usr/bin/python

import sys
import os
import csv
import glob

SPLIT_CHAR = '\\'
MAX_IMG_PER_CLASS = 50000

def whereAmI():
    return os.path.dirname(os.path.realpath(__import__("__main__").__file__))

path = sys.argv[1] # path to folder with right subfolder-class structure 
#absolutePath = whereAmI() + "/" + path
absolutePath = path

directories = [x[0] for x in os.walk(absolutePath)]

data = []

for directory in directories:
    tag = directory.rsplit(SPLIT_CHAR, 1)[-1]
    images = glob.glob(directory + "/*.jpg")

    if len(images) > 0:
        img_Cnt = 0
        for image in images:
            if img_Cnt <= MAX_IMG_PER_CLASS:
              taggedImage = [image, tag]
              data.append(taggedImage)
              print(taggedImage)
              img_Cnt += 1
            else:
              print("Got {0} images, breaking".format(img_Cnt))
              break

with open("images.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerows(data)

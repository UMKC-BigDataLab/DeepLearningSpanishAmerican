"""
Created on %(April 2019)
@author: %(Nouf Arasheed)
3x3 image bluring is used to remove background noise.
Modified on June 29, 2021 by Shivika Prasanna.
"""

import os
import sys
import numpy as np
from cv2 import cv2
import random
from imutils import paths

# pages = os.listdir("/content/Rollo38")
input_path = ("/content/Rollo38")
output_path = ("/content/Rollo38")

maxValue = 255
#input_path = sys.argv[1]
imagePaths = sorted(list(paths.list_images((input_path))))
print(imagePaths)
# random.seed(42)
#random.sorted(imagePaths)
random.shuffle(imagePaths)
for imagePath in imagePaths:
    head, tail = os.path.split(imagePath)
    print(tail)
    img = cv2.imread(imagePath)
    blur = cv2.blur(img, (3, 3))
    #outputPath = sys.argv[2]
    th, dst = cv2.threshold(blur, 127, maxValue, cv2.THRESH_BINARY)
    cv2.imwrite(output_path+tail, dst)

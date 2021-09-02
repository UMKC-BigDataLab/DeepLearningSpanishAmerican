# Improting Image class from PIL module
from PIL import Image
import csv
import random
import sys
import os
pages = ['90','91']
inpath = "/home/nouf/DL_Spanish/words/de_Cray/test/"
csv_path = "out/csv/"
outpath = "yolo-v3-words/"



for p in pages:
    name = os.path.splitext(p)[0]
    print(name)   
    directory1= outpath+name
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    folder = directory1 
    
    im = Image.open(inpath+name+".jpg")
    width, height = im.size
    with open(csv_path+name+".txt") as f:
        cf = csv.reader(f,delimiter =' ')
        n = 1
        for row in cf:
            
            img_id = str(row[0])
            print(img_id)
            left=int(row[1])
            top=int(row[2])
            right=int(row[3])
            bottom=int(row[4])
            im1 = im.crop((left, top, right, bottom))
            print(outpath+img_id)
            im1.save(outpath + img_id,"PNG")
            n+=1

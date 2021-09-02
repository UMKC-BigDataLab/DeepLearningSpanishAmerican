import numpy as np
import cv2
import os
import pandas as pd
import string
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image
import csv
import random
import sys
import os
import json
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras_tqdm import TQDMNotebookCallback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

char_list = "=!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcçdefghijklmnñopqrstuvwxyz"

def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape

#     _, img = cv2.threshold(img,
#                            128,
#                            255,
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Aspect Ratio Calculation
    new_w = 64
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape

    img = img.astype('float32')

    # Converts each to (32, 128, 1)
    if w < 64:
        add_zeros = np.full((64-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 64:
        dim = (128,64)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    # Normalize
    img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img

inputs = Input(shape=(64,128,1))
 
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)

# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)

# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
print(conv_7.shape) 
#squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
inner = Reshape(target_shape=(32,1488), name='reshape')(conv_7)
squeezed = Dense(64, activation='relu', name='dense1')(inner) 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
blstm_3 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_2) 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_3)

# model to be used at test time
act_model = Model(inputs, outputs)
act_model.summary()

# load the saved best model weights
act_model.load_weights('/mnt/HDDrive/nouf/YOLO-CRNN/crnn/Model/model-adamo-22463r-1000e-20199t-2246v.hdf5')

#PREDICT

files_path = os.listdir("out/csv/")
input_path = "out/csv/"
words_path = "yolo-v3-words/"
out_path = 'yolo-v3-predictions/'

for csv_file in files_path:
    page_num = os.path.splitext(csv_file)[0]
    
    full_path = input_path + csv_file
    test_file = open(full_path , "r")
    test = csv.reader(test_file,delimiter=' ')
    
    for line in test:
        head, tail = os.path.split(line[0])
        img_path = words_path + line[0]
        #img_path = words_path+page_num+"/"+line[0]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = process_image(img)
        img=np.expand_dims(img, axis=0)
        prediction =act_model.predict(img)

        # use CTC decoder
        decoded = K.ctc_decode(prediction,
                       input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                       greedy=True)[0][0]

        out = K.get_value(decoded)

        out_file = open(out_path+page_num+".txt", "a")
        writer = csv.writer(out_file, delimiter=' ')

        for i, x in enumerate(out):
        #print("original_text =  ", train_original_text[542+i])
        #print("predicted text = ", end = '')
            word = ''
            for p in x:
                if int(p) != -1:
                    word =word + char_list[int(p)]
            #print(char_list[int(p)], end = '')
            #word = word.replace('xx','')
            #print (name + word)
            writer.writerow([line[0]  , line[1] , line[2] , line[3] , line[4], str(word) ])
            
    print(csv_file + "  Done!")

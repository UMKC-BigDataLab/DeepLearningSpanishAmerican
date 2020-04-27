import matplotlib
import time
matplotlib.use("Agg")
from matplotlib import ticker
from glob import glob
from keras import backend as K
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
from tensorflow import keras
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.applications.nasnet import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split


#from utils import get_id_from_file_path, data_gen, chunker
#used packages
from datetime import datetime
#from packaging import version
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from imutils import paths
import scipy
import scipy.misc
from scipy.misc import imresize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import random
import pickle
import cv2
import os
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

#how to run:
# python3 trainvgg16.py --dataset /path/to/data --weights vgg16.h5 --model vgg16.model --label-bin vgg16.pickle --classes 19
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to output model")
ap.add_argument("-w","--weights",required=True, help="path to model weights")
ap.add_argument("-l", "--labels", required=True, help="path to output label binarizer")
#ap.a dd_argument("-p", "--plot", default="plot.png", help="path to output accuracy/loss plot")
ap.add_argument("-t","--tensorboard",required=True, help="path to tensorboard logs")
ap.add_argument("-c","--classes", required=True, help="Number of classes")
ap.add_argument("-v","--version", required=True, help="experiment version")
ap.add_argument("-n","--name", required=True, help="model name")
args = vars(ap.parse_args())
start_time = time.time()
print("[INFO] loading images...")
data = []
labels = []
NesteadPath =  (args["dataset"]) + "/../"


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(((args["dataset"])))))
#(list(paths.list_images('dataset/**/*.jpg'))
random.seed(42)
random.shuffle(imagePaths)
IMG_SIZE= (200,200)
# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image_pixels = cv2.imread(imagePath)
	resized_image_pixels = cv2.resize(image_pixels, IMG_SIZE)
	data.append(resized_image_pixels)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
	#print(labels)
	print (imagePath)

# scale the raw pixel intensities to the range [0, 1]
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data)
labels = np.array(labels)
print ("shape of train images is :", data.shape)
print ("shape of labels is : ", labels.shape)


# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(x_train, x_test, y_train, y_test) = train_test_split(data,
	labels, test_size=0.20, random_state=42)


#get lenghth of train and test data
ntrain = len(x_train)
ntest = len(x_test)
print("There are {} train images and {} test images.".format(np.asarray(x_train).shape[0], np.asarray(x_test).shape[0]))
print('There are {} unique classes to predict.'.format(np.unique(y_train).shape[0]))
# convert the labels from integers to vectors 
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

from keras.applications import VGG16
from keras import models 
from keras import layers 
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
from keras import optimizers
from keras.applications.nasnet import NASNetMobile
from keras.layers import Dense, Input, Dropout, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam

tboard = str(args["tensorboard"])
logdir = tboard + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)



inputs = Input((200,200, 3))
conv_base= NASNetMobile(weights= None,include_top=False, input_shape=(200,200, 3))#, weights=None
conv_base.summary()
x = conv_base.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(int(args["classes"]),activation='sigmoid')(x)
model = Model(inputs=conv_base.input, outputs=predictions)
conv_base.trainable=False
model.summary()
h5_path = "Nasnet_bestmodel.h5"
checkpoint = ModelCheckpoint(h5_path, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
preds = []
ids = []

train_datagen = ImageDataGenerator(rescale = 1./255, #scale image between 0 and 1
                                rotation_range=20,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255) # we don augment validation data, we only perform rescale

train_batchsize = 32
val_batchsize=32
#create image generator
train_generator= train_datagen.flow(x_train,y_train, batch_size = train_batchsize)
val_generator = val_datagen.flow(x_test,y_test, batch_size= val_batchsize, shuffle=False)


model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit_generator(train_generator,
                        steps_per_epoch = ntrain/train_generator.batch_size,
                        epochs = 150,
                        validation_data=val_generator,
                        validation_steps= ntest/val_generator.batch_size,
                        verbose=1,callbacks=[checkpoint,tensorboard_callback])


# save the model and label binarizer to disk
print("Saving the model and labels")
model.save_weights(args["weights"])
model.save(args["model"])
l_name = str(args["labels"])
f = open(l_name, "wb")
f.write(pickle.dumps(lb))
f.close()
#print("Average Training Accuracy: %.2f%%" % avg_acc*100)
score = model.evaluate(x_test,y_test, verbose=0)
print(score)
ex_time = (time.time() - start_time)
print("--- %s seconds ---" % (time.time() - start_time))
# convert the history.history dict to a pandas DataFrame:
hist_df = pd.DataFrame(history.history)

version = str(args["version"])
modelname= str(args["name"])
# or save to csv:
hist_csv_file = modelname + "_"+ version +"_history.csv"
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
time_file = modelname + "_time.txt"
t = open(time_file,"a")
t.write(modelname + "_" + version +":  "+ str(ex_time))


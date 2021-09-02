'''
Developed by Shivika Prasanna on 05/17/2021.
Last updated on 05/20/2021.
Get predictions using Keras-OCR retrained weights.
Run in terminal as:  python3 predictions.py -i <images_path> -m <model_path>
'''


import numpy as np
import pandas as pd
import json

import os.path
import argparse

import matplotlib.pyplot as pyplot
import keras_ocr

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="model path")
ap.add_argument("-i", "--test_img", required=True, help="test images path")
args = vars(ap.parse_args())

model_path = str(args["model"])
test_images_path = str(args["test_img"])

DEFAULT_ALPHABET = ''.join(['"', '=', 'C', 'D', 'J', 'N', 'R', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z', 'ç', 'ñ', 'ü'])

detector = keras_ocr.detection.Detector()
recognizer = keras_ocr.recognition.Recognizer(alphabet=DEFAULT_ALPHABET)

recognizer.model.load_weights(model_path)

path = [os.path.join(test_images_path, filename) for filename in os.listdir(test_images_path) if filename.endswith('.jpg')]

for images in path:
    image_result = {}
    tail = os.path.basename(images)
    json_filepath = tail.replace('.jpg', '.json')
    image = keras_ocr.tools.read(images)

    boxes = detector.detect(images=[image])[0]
    predictions = recognizer.recognize_from_boxes(images=[image], box_groups=[boxes.tolist()])

    image_result["filename"] = tail
    image_result["output"] = []

    for index, item in enumerate(zip(boxes.tolist(), predictions[0])):
        image_result["output"].append({"prediction": item[1], "box": item[0]})

    json.dump(image_result, open(json_filepath, "w"))

print("Done!")

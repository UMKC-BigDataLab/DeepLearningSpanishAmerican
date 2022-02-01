'''
Developed by Shivika Prasanna on 05/17/2021.
Last updated on 05/20/2021.
Find the overlap for each set of <ground truth, prediction> bounding boxes using Intersection over Union.
Run in terminal as:  python3 IoU.py -p <file.json> -g <file.csv> -i <image.jpg> -o <output directory> -t <0/1>
'''

# If using Google Colab, uncomment line 9.
# from google.colab.patches import cv2_imshow

import numpy as np
import pandas as pd
import json

import os.path
import argparse

from collections import namedtuple

from cv2 import cv2
from matplotlib import pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-g", "--ground", required=True, help="ground truth in .csv file path")
ap.add_argument("-p", "--pred", required=True, help="predictions in .json file path")
ap.add_argument("-i", "--image", required=True, help="image file path")
ap.add_argument("-o", "--output", required=True, help="output directory")
ap.add_argument("-t", "--technique", required=True, choices=[0,1], type=int, default=0, help="technique-keras(0) or yolo(1) format for predictions")
args = vars(ap.parse_args())

if str(args["ground"]).endswith('.csv'):
    ground_truth_path = str(args["ground"])
else:
    print("Ground truth file is not in CSV format!")
    exit()

predictions_path = str(args["pred"])
image_path = str(args["image"])
technique_option = int(args["technique"])
dir = str(args["output"])
tail = os.path.basename(image_path).replace('.jpg', '')

print("dir: ", dir, "tail: ", tail)

df_GT = pd.read_csv(ground_truth_path, header=None)
df_GT_1 = df_GT
df_GT_words = df_GT_1[0].str.extract("png\s(.*)\s\d+\s\d+\s\d+\s\d+", expand=True)
df_GT = df_GT[0].str.extract("(\d+\s\d+\s\d+\s\d+)", expand=True)
df_bb_gt = df_GT[0].str.split(" ", n=4, expand=True)
df_bb_gt.columns = ['x1', 'x2', 'y1', 'y2']
df_bb_gt["gt"] = df_GT_words[0]

df_all = pd.DataFrame(columns=["gtbb", "predbb", "gt", "pred", "overlap"], index=[0, 1, 2, 3, 4])

bb1_list = []
bb2_list = []
for index, row in df_bb_gt.iterrows():
  bb1_list.append({"x1": int(row[0]), "x2":  int(row[1]), "y1": int(row[2]), "y2": int(row[3]), "gt": row[4]})

with open(predictions_path) as json_file:
  p = json.load(json_file)
  if 'output' in p:
    for item in p["output"]:
        prediction = item["prediction"]
        
        # Yolo format.
        if technique_option == 1:
          x1, x2, y1, y2 = item["box"]
          bb2_list.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2, "pred": prediction})
        # Keras format.
        else:
          row = item["box"]
          x1 = int(row[0][0])
          x2 = int(row[1][0])
          x3 = int(row[2][0])
          x4 = int(row[3][0])

          y1 = int(row[0][1])
          y2 = int(row[1][1])
          y3 = int(row[2][1])
          y4 = int(row[3][1])

          top_left_x = min([x1,x2,x3,x4])
          top_left_y = min([y1,y2,y3,y4])
          bot_right_x = max([x1,x2,x3,x4])
          bot_right_y = max([y1,y2,y3,y4])

          bb2_list.append({"x1": top_left_x, "x2": top_left_y, "y1": bot_right_x, "y2": bot_right_y, "pred": prediction})

# print("BB1: ", bb1_list)
# print("BB2: ", bb2_list)

Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

overlaps = [Detection(image_path, bb1_list, bb2_list)]

counter = 0
for detection in overlaps:
  image = cv2.imread(detection.image_path)

  for gt_row in bb1_list:
    for pred_row in bb2_list:
      iou = bb_intersection_over_union(list(gt_row.values()), list(pred_row.values()))

      if iou >= 0.5:
        img = cv2.rectangle(
            image, (gt_row['x1'], gt_row['x2']), (gt_row['y1'], gt_row['y2']), (0, 255, 0), 2)
        img = cv2.rectangle(
            image, (pred_row['x1'], pred_row['x2']), (pred_row['y1'], pred_row['y2']), (0, 0, 255), 2)
        cv2.putText(img, "IoU: {:.4f}".format(iou), (gt_row['x1'], gt_row['x2']), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        df_all.at[counter, "gtbb"] = [gt_row['x1'],
                                      gt_row['x2'], gt_row['y1'], gt_row['y2']]
        df_all.at[counter, "predbb"] = [pred_row['x1'],
                                        pred_row['x2'], pred_row['y1'], pred_row['y2']]
        df_all.at[counter, "gt"] = gt_row["gt"]
        df_all.at[counter, "pred"] = pred_row["pred"]
        df_all.at[counter, "overlap"] = iou

        counter += 1

  cv2.imshow(tail + '-iou', image)
  cv2.imwrite(dir + "/" + tail + '-iou' + '.jpg', image)

df_compare = df_all[df_all['gt'] == df_all['pred']]

print("Number of perfect hits where GT matches Prediction: ", len(df_compare))

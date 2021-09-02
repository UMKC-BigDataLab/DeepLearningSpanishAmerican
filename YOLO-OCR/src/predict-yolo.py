import argparse
from PIL import Image
import cv2
import numpy as np
import csv
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--folder", default='/home/nouf/DL_Spanish/words/de_Cray/test/', help="image for prediction")
parser.add_argument("--config", default='/mnt/HDDrive/nouf/YOLO-CRNN/darknet/words/cfg/yolov3-custom.cfg', help="YOLO config path")
parser.add_argument("--weights", default='/mnt/HDDrive/nouf/YOLO-CRNN/darknet/backup-oneclass/yolov3-custom_190000.weights', help="YOLO weights path")
parser.add_argument("--names", default='/mnt/HDDrive/nouf/YOLO-CRNN/darknet/words/classes.names', help="class names path")
args = parser.parse_args()



CONF_THRESH, NMS_THRESH = 0.00, 0.04
page = str(args.folder)
pages = os.listdir(page)
print (pages)
out_path = "out/"
# Load the network
net = cv2.dnn.readNetFromDarknet(args.config, args.weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for i in pages:

# Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #path = "Rollo_38/"
    im = page+ i
    print(im)
#img = image = Image.open(page)
    img = cv2.imread(im)
    height, width , c = img.shape
    #print(img)
#width, height  = img.size

    blob = cv2.dnn.blobFromImage(img, 0.00392, (608,608), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                b_boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

    print(confidences)

    indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH).flatten().tolist()

# Draw the filtered bounding boxes with their class to the image
    with open(args.names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    head, t = os.path.splitext(i)
    boxes_path = open(out_path +"csv/"+ str(head) + ".txt" , "w")
    boxes_file = csv.writer(boxes_path, delimiter=' ')
    n = 1
    for index in indices:
        x, y, w, h = b_boxes[index]
        img_id = (str(n)+".png")
        boxes_file.writerow([str(head) + "/" + img_id ,x, y, x+w , y+h ])
        n+=1
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
        cv2.putText(img, str(head) + "/" + img_id , (x + 5, y ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 2)
    cv2.imwrite( out_path +"imgs/" + i , img)

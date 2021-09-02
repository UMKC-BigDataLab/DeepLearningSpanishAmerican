# YOLO-OCR: 
is an innovative model that combines the power of YOLO and CRNN-based text recognizer to detect and recognize words in historical Spanish manuscripts.

## To train YOLO-OCR
1. Train the CRNN text recognizer
```
python3 train-crnn.py --dataset /path/to/dataset/ --labels /path/to/labels/file/ --model /path/to/save/the/model/ --epoch "number of epochs" --batch "batch size"
```
2. Train YOLO to detect the words location

   a. Clone the YOLOv3-Darknet Repository
``` git clone  https://github.com/AlexeyAB/darknet.git ```

   b. Copy YOLO/words to darknet 

   c. Start training yolo ``` ./darknet detector -i 0 train words/detector.data words/cfg/yolov3-custom.cfg ```

## To predict words using YOLO-OCR
1. Run ``` python3 src/predict-yolo-ocr.py ```

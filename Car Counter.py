import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from sort import *

#Traffic_Video_Path
Video_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\Highway_Video.mp4'

#Covered Area Image Path
coveredArea_Image_path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\Frame_for_HighWay.png'

#Car Counter Image Path
Counter_Img_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\Vehicle_Count.png'

def convert_path(path):
    path = path.replace(r"C:\\", "/")
    path = path.replace("\\", "/")
    return path

cap = cv2.VideoCapture(convert_path(Video_Path))
Covered_Area = cv2.imread(convert_path(coveredArea_Image_path))
car_image = convert_path(Counter_Img_Path)

#YOLO large model path
YOLO_Large_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\yolov8l.pt'

#YOLO Nano model path
YOLO_Nano_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\yolov8n.pt'

model = YOLO(convert_path(YOLO_Nano_Path))

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag" "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
              "microwave", "cell phone", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

#Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
Counter_Line = [10, 600, 1800, 600]
total_count = []

#Image Parameters
Width = 1080
Height = 720

while True:
    _, img = cap.read()
    imgRegion = cv2.bitwise_and(img, Covered_Area)              #To capture only essential area
    results = model(imgRegion, stream=True)

    #Insert Car Counter Image
    imgCarGraphics = cv2.imread(car_image, cv2.IMREAD_UNCHANGED)
    resized_image = cv2.resize(imgCarGraphics, (400, 100), interpolation=cv2.INTER_AREA)
    img = cvzone.overlayPNG(img, resized_image, (10,10))

    #Detection for tracker
    detection = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            #Put the class name
            conf = math.ceil(box.conf[0]*100)/100
            cls = int(box.cls[0])
            current_ClassName = classNames[cls]

            if (current_ClassName == 'car' or current_ClassName == 'bus' or current_ClassName == 'truck') and conf>0.3:

                cvzone.putTextRect(img, f'{current_ClassName} {conf}', (max(0, x1), max(35, y1)), scale=2,
                                   thickness=2, colorR=(227, 126, 18), colorT=(255, 255, 255))

                currentArray = np.array([x1,y1,x2,y2,conf])
                detection = np.vstack((detection, currentArray))


    results_tracker = tracker.update(detection)
    cv2.line(img, ((Counter_Line[0], Counter_Line[1])), ((Counter_Line[2], Counter_Line[3])), (0, 0, 255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))

        cx, cy = x1+w//2, y1+h//2

        if Counter_Line[0] < cx < Counter_Line[2] and Counter_Line[1] - 20 <cy < Counter_Line[1] + 20:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img, ((Counter_Line[0], Counter_Line[1])), ((Counter_Line[2], Counter_Line[3])), (0, 255, 0), 5)


    cv2.putText(img, "Counter Line", (660, 575), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 252, 160), 6, cv2.LINE_AA)
    cv2.putText(img, ':', (290, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(img, str(len(total_count)), (310, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)


    img = cv2.resize(img, (Width, Height))
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

    #cv2.imshow('Image2', imgRegion)  #to show with covered area
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()

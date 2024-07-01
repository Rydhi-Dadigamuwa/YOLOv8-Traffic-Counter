# YOLOv8-Traffic-Counter

This project counts vehicles traveling on a road using the YOLOv8 object detection model.

## Features
1. Real-time vehicle detection with bounding boxes and counting.
2. Optimized for GPU acceleration using CUDA and cuDNN.


## Notes
1. This project uses YOLOv8 for object detection.
2. Recomended to open in Pycharm.
3. Make sure your system has a compatible NVIDIA GPU and the necessary drivers for CUDA and cuDNN. This is very helpful, if you expect faster responses.
4. Update the paths for following variables in your script to relevant local paths on your machine:
 ```python
    Video_Path = r'C:\path\to\your\Highway_Video.mp4'
     coveredArea_Image_path = r'C:\path\to\your\Frame_for_HighWay.png'
    Counter_Img_Path = r'C:\path\to\your\Vehicle_Count.png'
    YOLO_Large_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\yolov8l.pt'
    YOLO_Nano_Path = r'C:\Users\DELL\PycharmProjects\Object-Detection-YOLOV8\Car Counter\YOLOv8-Traffic-Counter\yolov8n.pt'
```


## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/vehicle-counter-yolov8.git
```

2. Install dependancies
```bash
pip install -r requirements.txt
```


 

from ultralytics import YOLO

model = YOLO('yolov8x-seg.pt')
result=model(source="cricket.mp4",show=True,conf=0.4,save=True)
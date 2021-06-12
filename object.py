import cv2
import numpy as np
net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes=[]
with open('coco.names','r') as f:
    classes=f.read().splitlines()
print(classes)
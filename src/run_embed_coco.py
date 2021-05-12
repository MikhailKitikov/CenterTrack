from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import json
import copy
import numpy as np

import sys
sys.path.insert(0, "/content/CenterTrack/src")
sys.path.insert(0, "/content/CenterTrack/src/lib")
sys.path.insert(0, "/content/CenterTrack/src/lib/model/networks/DCNv2")

import _init_paths
from opts import opts
from detector import Detector


if __name__ == '__main__':

  coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush']

  MODEL_PATH = "/content/CenterTrack/models/coco_tracking.pth"
  TASK = "tracking"
  opt = opts().init("{} --load_model {}".format(TASK, MODEL_PATH).split(' '))
  detector = Detector(opt)

  import cv2
  name = '00000.ppm'
#   img = cv2.imread(f'/content/CenterTrack/samples/images/{name}')

  img = cv2.imread(f'/content/CenterTrack/samples/images/FullIJCNN2013/{name}')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  import time
  start_time = time.time()
  results = detector.run(img)['results']
  end_time = time.time()
  print('time:', end_time - start_time)
  
  print("\nResults:")
  for res in results:
    print(res)
    bbox = res['bbox']
    start_point = (int(bbox[0]), int(bbox[1]))
    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = (int(bbox[2]), int(bbox[3]))
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 1
    
    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    img = cv2.rectangle(img, start_point, end_point, color, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (int(bbox[0]), int(bbox[1] - 10))
    
    # fontScale
    fontScale = 0.5
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 1
    
    # Using cv2.putText() method
    img = cv2.putText(img, coco_classes[res['class'] - 1], org, font, fontScale, color, thickness, cv2.LINE_AA)

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  cv2.imwrite(f'/content/{name}', img)

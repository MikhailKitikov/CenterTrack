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

  MODEL_PATH = "/content/CenterTrack/models/nuScenes_3Dtracking.pth"
  TASK = "tracking,ddd"
  opt = opts().init("{} --load_model {}".format(TASK, MODEL_PATH).split(' '))
  detector = Detector(opt)

  import cv2
  img = cv2.imread('/content/CenterTrack/samples/images/frame.jpg')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  results = detector.run(img)['results']
  print("\nResults:")
  for res in results:
    print(res)

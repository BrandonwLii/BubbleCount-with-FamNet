""" 
    This script is used for generating the bounding box files of all exemplar images
    in the specified directory
"""

import os
import cv2
from utils import select_exemplar_rois

#### Specify exemplar folder ###################################################
dir_path = "G:\\My Drive\\ECE MEng Courses\\ECE2500\\LearningToCountEverything\\Exemplars"
################################################################################

file_list = os.listdir(dir_path)
for file in file_list:
    if not file.lower().endswith('.jpg'):
        continue
    
    print(f"Selecting exemplars from {file}...")
    image_path = os.path.join(dir_path, file)
    out_bbox_file = f"{image_path}_boxx.txt"
    fout = open(out_bbox_file, "w")

    im = cv2.imread(image_path)
    cv2.imshow('image', im)
    rects = select_exemplar_rois(im)

    coord_reorder = list()
    for rect in rects:
        y1, x1, y2, x2 = rect
        coord_reorder.append([y1, x1, y2, x2])
        fout.write("{} {} {} {}\n".format(y1, x1, y2, x2))

    if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) >= 0:
        cv2.destroyWindow("image")

    print(f"selected bounding boxes are saved to {out_bbox_file}")
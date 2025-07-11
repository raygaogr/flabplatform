import json
import os
import numpy as np
from copy import deepcopy
import cv2


if __name__ == "__main__":
    img_dir = "datasets/coco8-seg/val/image"
    label_dir = "datasets/coco8-seg/val/label"
    pred_dir = "res/yolov8/segment/label"

    eval_file_name = "000000000061"

    img = cv2.imread(os.path.join(img_dir, eval_file_name+".jpg"))
    pred_img = deepcopy(img)

    with open(os.path.join(label_dir, eval_file_name+".json"), "r") as f:
        label = json.load(f)

    with open(os.path.join(pred_dir, eval_file_name+".json"), "r") as f:
        pred = json.load(f)


    for shape in label["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        label_res = cv2.drawContours(img, [points], -1, (0, 255, 0), 2)
    cv2.imwrite("label.jpg", label_res)

    for shape in pred["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        pred_res = cv2.drawContours(pred_img, [points], -1, (0, 0, 255), 2)
    cv2.imwrite("pred.jpg", pred_res)




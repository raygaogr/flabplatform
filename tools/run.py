import json
import os.path as osp
import numpy as np
from copy import deepcopy
import cv2
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
from flabplatform.core.config import Config

if __name__ == "__main__":
    cfg = Config(
    dict(pipeline=[dict(type='LoadImage'),
        dict(type='LoadAnnotations')]))

    options = {"pipeline.0.type": "dfsfs"}

    cfg.merge_from_dict(options)

    print(cfg)




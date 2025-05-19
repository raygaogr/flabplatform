import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
from flabplatform.flabdet.models import attempt_load_one_weight


if __name__ == '__main__':
    model, ckpt = attempt_load_one_weight("res/detect/train/weights/best.pt")
    print(model)

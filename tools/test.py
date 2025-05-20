import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(__file__)))
# from flabplatform.flabdet.models import attempt_load_one_weight
import torch

if __name__ == '__main__':
    # model, ckpt = attempt_load_one_weight("res/detect/train/weights/best.pt")
    # ckpt1 = torch.load(r"D:\Workspace_gr\pyProjects\flabplatform\models\yolo11n.pt", map_location="cpu")
    ckpt1 = torch.load(r"D:\Workspace_gr\pyProjects\flabplatform\models\yolo11n.pt", map_location={"ultralytics.nn.tasks": "flabplatform.flabdet.models.yolomodel"})
    ckpt2 = torch.load(r"D:\Workspace_gr\pyProjects\flabplatform\res\detect\train\weights\best.pt", map_location="cpu")
    ckpt3 = torch.load(r"D:\Workspace_gr\pyProjects\flabplatform\res\detect\train1\weights\best.pt", map_location="cpu")
    print(ckpt1)

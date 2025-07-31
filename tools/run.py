import torch
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
if __name__ == "__main__":
    a = torch.load("D:/Workspace_gr/pyProjects/flabplatform/models/dab_detr/best.pt", map_location="cpu")




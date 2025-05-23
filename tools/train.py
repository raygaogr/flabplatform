import argparse
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
import yaml
import json
import torch
import os


def set_environ(device="", batch=0 ):
    device = str(device).lower()
    for remove in "cuda:", "none", "(", ")", "[", "]", "'", " ":
        device = device.replace(remove, "")  # to string, 'cuda:0' -> '0' and '(0, 1)' -> '0,1'
    cpu = device == "cpu"
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        if device == "cuda":
            device = "0"
        if "," in device:
            device = ",".join([x for x in device.split(",") if x])  # remove sequential commas, i.e. "0,,1" -> "0,1"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        if not (torch.cuda.is_available() and torch.cuda.device_count() >= len(device.split(","))):
            install = (
                "See https://pytorch.org/get-started/locally/ for up-to-date torch install instructions if no "
                "CUDA devices are seen by torch.\n"
                if torch.cuda.device_count() == 0
                else ""
            )
            raise ValueError(
                f"Invalid CUDA 'device={device}' requested."
                f" Use 'device=cpu' or pass valid CUDA device(s) if available,"
                f" i.e. 'device=0' or 'device=0,1,2,3' for Multi-GPU.\n"
                f"\ntorch.cuda.is_available(): {torch.cuda.is_available()}"
                f"\ntorch.cuda.device_count(): {torch.cuda.device_count()}"
                f"\nos.environ['CUDA_VISIBLE_DEVICES']: {visible}\n"
                f"{install}"
            )

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='The yaml config file path')
    args = parser.parse_args()
    return args

def main():
    # 这里可以改成自己的配置文件路径
    args = parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        if 'yaml' in args.config:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        elif 'json' in args.config:
            cfg = json.load(f)
        else:
            raise ValueError("config file must be yaml or json")
    device = cfg["training"]["algoParams"]["device"]
    set_environ(device)



    from flabplatform.core.engine import create_runner
    # 构建执行器
    runner = create_runner(args)

    # 训练：
    runner.train()

    # 验证评估：
    val_res = runner.val(imgsz=640, batch=16, conf=0.25, iou=0.6)
    print(val_res)

    # 预测：
    pred_res = runner.predict(source="assets/bus.jpg", conf=0.25, iou=0.6)
    # or
    pred_res = runner("assets/bus.jpg", conf=0.25, iou=0.6)
    for res in pred_res:
        res.save("busres.jpg")

    # 导出：
    runner.export(format="onnx")



if __name__ == '__main__':
    main()
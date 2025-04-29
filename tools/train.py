import argparse
import os
import os.path as osp
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
from flabplatform.core.engine import create_runner

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    # 这里可以改成自己的配置文件路径
    args = parse_args()

    # 构建执行器
    runner = create_runner(args)


    # 训练：
    runner.train()

    # 验证评估：
    val_res = runner.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)
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
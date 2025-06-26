# classification

## 源码打包方式：

`python setup.py bdist_wheel` , 得到 whl 包 fdcv-0.0.1-py3-none-any.whl

## 安装方式：

`pip install  fdcv-0.0.1-py3-none-any.whl`

## 调用方式示例：

参见 example 下脚本

### 训练:

`python train.py classification_config.yaml`

### 评估：

`python eval.py classification_config.yaml --eval --export_onnx --onnx_eval`

### onnxruntime 推理：

`python onnx_inference.py --config classification_onnx_inference.yaml --image_paths /root/dataset/tiny-imagenet-200/val/n01443537/val_1551.JPEG`


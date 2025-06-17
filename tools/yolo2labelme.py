import os
import glob
import numpy as np
import cv2
import json

# 可以将yolov8目标检测生成的txt格式的标注转为json，可以使用labelme查看标注
# 该方法可以用于辅助数据标注
def convert_txt_to_labelme_json(txt_path, image_path, output_dir, class_name, image_fmt='.jpg', is_detect=True):
    """
    将文本文件转换为LabelMe格式的JSON文件。
    此函数处理文本文件中的数据，将其转换成LabelMe标注工具使用的JSON格式。包括读取图像，
    解析文本文件中的标注信息，并生成相应的JSON文件。
    :param txt_path: 文本文件所在的路径
    :param image_path: 图像文件所在的路径
    :param output_dir: 输出JSON文件的目录
    :param class_name: 类别名称列表，索引对应类别ID
    :param image_fmt: 图像文件格式，默认为'.jpg'
    :return:
    """
    # 获取所有文本文件路径
    txts = glob.glob(os.path.join(txt_path, "*.txt"))
    for txt in txts:
        # 初始化LabelMe JSON结构
        labelme_json = {
            'version': '5.5.0',
            'flags': {},
            'shapes': [],
            'imagePath': None,
            'imageData': None,
            'imageHeight': None,
            'imageWidth': None,
        }
        # 获取文本文件名
        txt_name = os.path.basename(txt)
        # 根据文本文件名生成对应的图像文件名
        image_name = txt_name.split(".")[0] + image_fmt
        labelme_json['imagePath'] = image_name
        # 构造完整图像路径
        image_name = os.path.join(image_path, image_name)
        # 检查图像文件是否存在，如果不存在则抛出异常
        if not os.path.exists(image_name):
            raise Exception('txt 文件={},找不到对应的图像={}'.format(txt, image_name))
        # 读取图像
        image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        # 获取图像高度和宽度
        h, w = image.shape[:2]
        labelme_json['imageHeight'] = h
        labelme_json['imageWidth'] = w
        # 读取文本文件内容
        with open(txt, 'r') as t:
            lines = t.readlines()
            for line in lines:
                point_list = []
                content = line.strip().split(' ')
                # 根据类别ID获取标签名称
                label = class_name[int(content[0])]  # 标签
                if is_detect:
                    # 解析点坐标
                    for index in range(1, len(content)):
                        if index == 1:  # 中心点归一化的x坐标
                            cx = float(content[index])
                        if index == 2:  # 中心点归一化的y坐标
                            cy = float(content[index])
                        if index == 3:  # 归一化的目标框宽度
                            wi = float(content[index])
                        if index == 4:  # 归一化的目标框高度
                            hi = float(content[index])
                    x1 = (2 * cx * w - w * wi) / 2
                    x2 = (w * wi + 2 * cx * w) / 2
                    y1 = (2 * cy * h - h * hi) / 2
                    y2 = (h * hi + 2 * cy * h) / 2
                    point_list = [[x1, y1], [x2, y2]]
                else:
                    for i in range(1, len(content), 2):
                        if i + 1 < len(content):
                            x = float(content[i]) * w
                            y = float(content[i + 1]) * h
                            point_list.append([x, y])
                # 构造shape字典
                shape = {
                    'label': label,
                    'points': point_list,
                    'group_id': None,
                    'description': None,
                    'shape_type': 'rectangle',
                    'flags': {},
                    'mask': None
                }
                if not is_detect:
                    shape["shape_type"] = "polygon"
                labelme_json['shapes'].append(shape)
            # 生成JSON文件名
            json_name = txt_name.split('.')[0] + '.json'
            json_name_path = os.path.join(output_dir, json_name)
            # 写入JSON文件
            fd = open(json_name_path, 'w')
            json.dump(labelme_json, fd, indent=2)
            fd.close()
            # 输出保存信息
            print("save json={}".format(json_name_path))
 
if __name__ == '__main__':
    txt_path = 'datasets/coco8-seg/labels/train'
    image_path = 'datasets/coco8-seg/images/train'
    output_dir = 'datasets/coco8-seg/labels/train'
    
    # 标签列表
    yolo_class_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush']  # 标签类别名
    convert_txt_to_labelme_json(txt_path, image_path, output_dir, yolo_class_name, is_detect=False)
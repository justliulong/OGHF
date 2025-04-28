'''
可保持yolo格式的数据集的训练集和验证集划分不变的同时转为coco数据集格式
原本的文件夹内存储的格式为：
yolo-dataset
├──images
│   ├──train
│   │   ├── imaget.png|jpg
│   │   └── *.png|jpg
│   └──val
│       ├── imagev.png|jpg
│       └── *.png|jpg
└──labels
    ├──train
    │   ├── imaget.txt
    │   └── *.txt
    └──val
        ├── imagev.txt
        └── *.txt
转换结果：
output
├── annotations
│   ├── train.json
│   └── val.json
├── visualizations # 可视化掩码，这是绘制在原图上的掩码，方便展示查验效果，通过添加“--v”参数来创建
│   ├──train
│   │   ├── imaget.png|jpg
│   │   └── *.png|jpg
│   └──val
│       ├── imagev.png|jpg
│       └── *.png|jpg
├── <train> # 脚本并不会创建这个存储图像的文件夹，需要从yolo数据集中手动拷贝过来
│   ├── imaget.png
│   └── *.png
└── <val> # 脚本并不会创建这个存储图像的文件夹，需要从yolo数据集中手动拷贝过来
    ├── imagev.png
    └── *.png

其中的数据集内容由如下的yolo格式:
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
<class-index> 是对象的类的索引，而 <x1> <y1> <x2> <y2> ... <xn> <yn> 是对象的分割掩码的边界坐标。坐标之间用空格隔开
转为如下的coco格式(以train.json为例，val.json同理):
{
    "info": {},
    "licenses": [],
    "images": [
        {
            "id": 1,
            "file_name": "imagev1.jpg",
            "width": 640,
            "height": 480,
            "license": 0,
            "date_captured": "XXXX-XX-XX XX:XX:XX"
        },
        {
            "id": 2,
            "file_name": "imagev2.jpg",
            "width": 640,
            "height": 480,
            "license": 0,
            "date_captured": "XXXX-XX-XX XX:XX:XX"
        },
        ...
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": <class-index>,
            "segmentation": [[<x1> <y1> <x2> <y2> ... <xn> <yn>]],
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": <class-index>,
            "segmentation": [[<x1> <y1> <x2> <y2> ... <xn> <yn>]],
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "area": area,
            "iscrowd": 0
        },
        ...
    ],
    "categories": [
        {
            "id": 0,
            "name": "cell",
            "supercategory": "animal"
        },
        ...
    ]
}
其中，"images"列表存储了所有图像的信息，"annotations"列表存储了所有标注信息，"categories"列表存储了所有类别信息。
"segmentation"字段存储了多边形点集，"bbox"字段存储了边界框，"area"字段存储了面积。

#? 使用方法：
提前在本脚本的入口函数(if __name__ == '__main__')中设置类别列表categories，格式为 [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}, ...]
然后运行如下命令：
python yolo2coco.py --i <yolo-dataset/images> --l <yolo-dataset/labels> --j <output/>
'''

import os
import cv2
import json
from tqdm import tqdm  # 用于显示进度条
import time
import numpy as np
import random
from pycocotools.coco import COCO


def yolo_to_coco(yolo_images_dir, yolo_labels_dir, output_json_path, categories):
    """
    将YOLO格式的实例分割数据集转换为COCO格式。

    参数:
    - yolo_images_dir: YOLO图像文件夹路径
    - yolo_labels_dir: YOLO标注文件夹路径
    - output_json_path: 输出的COCO格式JSON文件路径
    - categories: 类别列表，格式为 [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}, ...]
    """
    # 初始化COCO格式的字典
    coco_format = {
        "info": {
            "description": None,
            "version": "1.0",
            "year": time.strftime("%Y"),
            "contributor": None,
            "date_created": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "licenses": [{
            "url": None,
            "id": 0,
            "name": None
        }],
        "images": [],
        "annotations": [],
        "categories": categories
    }

    image_id = 1  # 图像ID从1开始
    annotation_id = 1  # 标注ID从1开始

    # 获取所有图像文件
    image_files = [f for f in os.listdir(yolo_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 遍历每张图像
    for image_file in tqdm(image_files, desc="Converting"):
        image_path = os.path.join(yolo_images_dir, image_file)
        annotation_path = os.path.join(
            yolo_labels_dir,
            image_file.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt"))

        # 读取图像尺寸
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue
        height, width, _ = image.shape

        # 添加图像信息到COCO格式
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height,
            "license": 0,
            "date_captured": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        # 读取对应的标注文件
        if os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                lines = f.readlines()

            # 处理每行标注
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:  # YOLO实例分割格式：class_id x1 y1 x2 y2 ... xn yn
                    print(f"Warning: Invalid annotation format in {annotation_path}. Skipping.")
                    continue

                class_id = int(parts[0])
                polygon = list(map(float, parts[1:]))  # 提取多边形点集

                # 将YOLO格式的归一化坐标转换为绝对坐标
                absolute_polygon = []
                for i in range(0, len(polygon), 2):
                    x = polygon[i] * width
                    y = polygon[i + 1] * height
                    absolute_polygon.extend([x, y])

                # 计算边界框
                x_coords = absolute_polygon[::2]
                y_coords = absolute_polygon[1::2]
                x_min = min(x_coords)
                y_min = min(y_coords)
                bbox_width = max(x_coords) - x_min
                bbox_height = max(y_coords) - y_min

                # 添加标注信息到COCO格式
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [absolute_polygon],  # COCO格式需要多边形点集
                    "bbox": [x_min, y_min, bbox_width, bbox_height],  # 边界框
                    "area": bbox_width * bbox_height,  # 面积
                    "iscrowd": 0  # 是否为群体
                })

                annotation_id += 1
        else:
            print(f"Warning: No annotation file found for {image_file}. Skipping.")

        image_id += 1

    # 保存COCO格式的JSON文件
    with open(output_json_path, "w") as f:
        json.dump(coco_format, f)

    print(f"Conversion complete! COCO format annotations saved to {output_json_path}")


def visualize_coco_annotations(coco_annotation_path, image_dir, output_dir):
    """
    使用OpenCV可视化COCO格式的实例分割标注并保存结果。

    参数:
    - coco_annotation_path: COCO格式的标注文件路径（JSON文件）
    - image_dir: 图像文件夹路径
    - output_dir: 保存可视化结果的文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 加载COCO标注文件
    coco = COCO(coco_annotation_path)

    # 获取所有图像的ID
    image_ids = coco.getImgIds()

    # 遍历每张图像
    for image_id in tqdm(image_ids, desc="Drawing annotations"):
        # 获取图像信息
        image_info = coco.loadImgs(image_id)[0]
        image_file = image_info["file_name"]
        image_path = os.path.join(image_dir, image_file)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_file}. Skipping.")
            continue

        # 获取该图像的标注
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        # 在图像上绘制标注
        for ann in annotations:
            # 为每个实例生成随机颜色
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            # 绘制多边形
            if "segmentation" in ann:
                for seg in ann["segmentation"]:
                    # 将点集转换为多边形
                    poly = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)  # 绘制多边形边界
                    cv2.fillPoly(image, [poly], color=color)  # 填充多边形内部（半透明）

            # 绘制边界框
            if "bbox" in ann:
                bbox = ann["bbox"]
                x, y, w, h = map(int, bbox)  # 转换为整数
                cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=2)  # 绘制边界框

            # 显示类别名称和实例ID
            if "category_id" in ann:
                category_id = ann["category_id"]
                category_name = coco.loadCats(category_id)[0]["name"]
                instance_id = ann["id"]  # 实例ID
                text = f"{category_name} ({instance_id})"
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # 白色文字

        # 保存可视化结果
        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, image)

        # print(f"Saved visualization for {image_file} to {output_path}")


def main(yolo_images_dir, yolo_labels_dir, output_json_dir, categories, visualize):
    res_json_dir = os.path.join(output_json_dir, 'annotations')
    if not os.path.exists(res_json_dir):
        os.makedirs(res_json_dir)
    # *********************************** train *******************************************
    print("converting train dataset...")
    trian_image_path = os.path.join(yolo_images_dir, 'train')
    train_label_path = os.path.join(yolo_labels_dir, 'train')
    res_json_file = os.path.join(res_json_dir, 'train.json')
    yolo_to_coco(trian_image_path, train_label_path, res_json_file, categories)
    # ************************************* visualize *****************************************
    if visualize:
        output_dir = os.path.join(output_json_dir, 'visualizations/train')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        visualize_coco_annotations(res_json_file, trian_image_path, output_dir)
    # ************************************ val ******************************************
    print("converting val dataset...")
    val_image_path = os.path.join(yolo_images_dir, 'val')
    val_label_path = os.path.join(yolo_labels_dir, 'val')
    res_json_file = os.path.join(res_json_dir, 'val.json')
    yolo_to_coco(val_image_path, val_label_path, res_json_file, categories)
    # *********************************** visualize *******************************************
    if visualize:
        output_dir = os.path.join(output_json_dir, 'visualizations/val')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        visualize_coco_annotations(res_json_file, val_image_path, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert YOLO format instance segmentation dataset to COCO format.")
    parser.add_argument("--i", type=str, required=True, help="Path to the directory containing YOLO images.")
    parser.add_argument("--l", type=str, required=True, help="Path to the directory containing YOLO labels.")
    parser.add_argument("--o", type=str, required=True, help="Path to the output coco dir.")
    parser.add_argument("--v", action="store_true", help="Visualize the converted dataset.")
    args = parser.parse_args()

    # 定义类别
    categories = [{"id": 0, "name": "cell"}]

    # 调用函数
    main(yolo_images_dir=args.i,
         yolo_labels_dir=args.l,
         output_json_dir=args.o,
         categories=categories,
         visualize=args.v)

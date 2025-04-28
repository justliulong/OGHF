'''
将labelme的数据集格式转为yolo数据集格式(不划分训练集和验证集)
原始labelme数据集目录结构(目录中有无图片文件都没关系)
json_dir:
├── image.json
├── image.jpg|png|jpeg
├── *.json
└── *.jpg|png|jpeg

转换后yolo数据集目录结构(脚本不会对图片进行复制或者移动)
labels_dir:
├── image.txt
└── *.txt


yolo格式中txt的存储如下:
目标检测:
<class-index> <x_center>, <y_center>, <w>, <h>
<class-index> 是对象的类的索引，而 <x_center>, <y_center>, <w>, <h> 分别是目标bbox的中心坐标x,中心坐标y,宽w,高h
实例分割：
<class-index> <x1> <y1> <x2> <y2> ... <xn> <yn>
<class-index> 是对象的类的索引，而 <x1> <y1> <x2> <y2> ... <xn> <yn> 是对象的分割掩码的边界坐标。坐标之间用空格隔开

#? 使用方法：
提前在本脚本的入口函数(if __name__ == '__main__')中设置类别列表class_name,具体格式为['class1', 'class2',...],
然后运行如下命令：
python labelme2yolo.py --i < json_dir > --o < labels_dir > < --d >

'''

import json
import os


def labelme2yolo_seg(class_name, json_dir, labels_dir):
    """
        此函数用来将labelme软件标注好的json格式转换为yolov_seg中使用的txt格式
        :param json_dir: labelme标注好的*.json文件所在文件夹
        :param labels_dir: 转换好后的*.txt保存文件夹
        :param class_name: 数据集中的类别标签
        :return:
    """
    list_labels = []  # 存放json文件的列表

    # 0.创建保存转换结果的文件夹
    if (not os.path.exists(labels_dir)):
        os.mkdir(labels_dir)

    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    for files in [f for f in os.listdir(json_dir) if f.endswith('.json')]:  # 遍历json文件夹下的所有json文件
        file = os.path.join(json_dir, files)  # 获取一个json文件
        list_labels.append(file)  # 将json文件名加入到列表中

    print(f"==>total json files: {len(list_labels)}")

    for labels in list_labels:  # 遍历所有json文件
        with open(labels, "r") as f:
            file_in = json.load(f)
            shapes = file_in["shapes"]
            print(labels)

        txt_filename = os.path.basename(labels).replace(".json", ".txt")
        txt_path = os.path.join(labels_dir, txt_filename)  # 使用labels_dir变量指定保存路径

        with open(txt_path, "w+") as file_handle:
            for shape in shapes:
                line_content = []  # 初始化一个空列表来存储每个形状的坐标信息
                line_content.append(str(class_name.index(shape['label'])))  # 添加类别索引
                # 添加坐标信息
                for point in shape["points"]:
                    x = point[0] / file_in["imageWidth"]
                    y = point[1] / file_in["imageHeight"]
                    line_content.append(str(x))
                    line_content.append(str(y))
                # 使用空格连接列表中的所有元素，并写入文件
                file_handle.write(" ".join(line_content) + "\n")


def labelme2yolo_det(class_name, json_dir, labels_dir):
    """
        此函数用来将labelme软件标注好的json格式转换为yolov_det中使用的txt格式
        :param json_dir: labelme标注好的*.json文件所在文件夹
        :param labels_dir: 转换好后的*.txt保存文件夹
        :param class_name: 数据集中的类别标签
        :return:
    """
    list_labels = []  # 存放json文件的列表

    # 0.创建保存转换结果的文件夹
    if (not os.path.exists(labels_dir)):
        os.mkdir(labels_dir)

    # 1.获取目录下所有的labelme标注好的Json文件，存入列表中
    for files in [f for f in os.listdir(json_dir) if f.endswith('.json')]:  # 遍历json文件夹下的所有json文件
        file = os.path.join(json_dir, files)  # 获取一个json文件
        list_labels.append(file)  # 将json文件名加入到列表中

    print(f"==>total json files: {len(list_labels)}")

    for labels in list_labels:  # 遍历所有json文件
        with open(labels, "r") as f:
            file_in = json.load(f)
            shapes = file_in["shapes"]
            # print(labels)

        txt_filename = os.path.basename(labels).replace(".json", ".txt")
        txt_path = os.path.join(labels_dir, txt_filename)  # 使用labels_dir变量指定保存路径

        with open(txt_path, "w+") as file_handle:
            for shape in shapes:
                line_content = []  # 初始化一个空列表来存储每个形状的坐标信息
                line_content.append(str(class_name.index(shape['label'])))  # 添加类别索引
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = shape['points']
                x1 = min(x1, x2, x3, x4)
                x2 = max(x1, x2, x3, x4)
                y1 = min(y1, y2, y3, y4)
                y2 = max(y1, y2, y3, y4)
                x1, x2 = x1 / file_in['imageWidth'], x2 / file_in['imageWidth']
                y1, y2 = y1 / file_in['imageHeight'], y2 / file_in['imageHeight']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # 中心点归一化的x坐标和y坐标
                wi, hi = abs(x2 - x1), abs(y2 - y1)  # 归一化的目标框宽度w，高度h
                line_content.append(str(cx))
                line_content.append(str(cy))
                line_content.append(str(wi))
                line_content.append(str(hi))
                # 使用空格连接列表中的所有元素，并写入文件
                file_handle.write(" ".join(line_content) + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='labelme2yolo')
    parser.add_argument('--i', type=str, default='data/labels', help='json文件所在文件夹')
    parser.add_argument('--o', type=str, default='data/labels_txt', help='保存txt文件所在文件夹')
    parser.add_argument('--d', action='store_true', help='任务类型，添加参数表示转换之后的结构是用于检测的，否则是分割的')

    args = parser.parse_args()
    # class_name = ['cell']
    class_name = ['table']

    if args.d:
        print("=============== detection ======================")
        labelme2yolo_det(class_name, args.i, args.o)
    else:
        print("=============== segment ======================")
        labelme2yolo_seg(class_name, args.i, args.o)

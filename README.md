<div id="top" align="center">

# OG-HFYOLO :Orientation Gradient Guidance and Heterogeneous Feature Fusion For Deformation Table Cell Instance Segmentation
  
  Long Liu, Cihui Yang* </br>
  
  [![arXiv](https://img.shields.io/badge/arXiv-2504.20682-b31b1b.svg)](https://arxiv.org/abs/2504.20682)

</div>

## Datasets
### Data generator
The data generator mentioned in the paper is specifically implemented in [./dataprocess/data_gen.py](./dataprocess/data_gen.py), and detailed usage instructions are provided at the beginning of the file.

Additionally, the [./dataprocess](./dataprocess) folder contains scripts that can convert various dataset formats, including ([yolo2coco](./dataprocess/yolo2coco.py), [yolo2labelme](./dataprocess/yolo2labelme.py), [labelme2yolo](./dataprocess/labelme2yolo.py), [labelme2coco](./dataprocess/labelme2coco.py)), with specific usage instructions detailed at the beginning of each file.

### Datasets
We store the dataset in *[DWTAL-s](https://drive.google.com/file/d/1i4meTuVevdtEUd7wde59Y7KzR27Dj9QF)* and *[DWTAL-l](https://drive.google.com/file/d/1wJiRt2u7sY9uqZxJtSJiWy_Zhu87yQOU)* by the YOLO dataset format, for example:
- DWTAL-s
    - image
        - train
            - image.jpg
            - ...
        - val
            - image.jpg
            - ...
    - label
        - train
            - label.txt
            - ...
        - val
            - label.txt
If you need other formats, you can use the scripts in the [./dataprocess](./dataprocess) folder for conversion.

Additionally, we have provided the labelme format for the corresponding dataset, which can be found under *[DWTAL-s.json.zip](https://drive.google.com/file/d/1_FJTwJWV3AwnaLHhpwGq9K4t-ywCfmV-)* and *[DWTAL-L.json.zip](https://drive.google.com/file/d/10H2oc_kQIlHyHXim4xSSrwDPivOciw-8)*. You can place the images and JSON labels in the same folder and use labelme to open the images and view the annotations.

Finally, we have also provided a dataset with logical coordinates corresponding to the DWTAL-l dataset, located in *[DWTAL-l.logical.zip](https://github.com/justliulong/OGHF/releases/download/v1.0.0/DTAL-l.local.zip)*, to advance the development of deformable table structure recognition.

## Model
The model structure mentioned in the paper is specifically implemented in [./cfg/models/segment/og-hfyolo.yaml](./cfg/models/segment/og-hfyolo.yaml).


### Environments
Install requirements.txt in a Python>=3.8.0 environment, including PyTorch>=1.8.
```
conda create -n py38 python=3.8
conda activate py38
pip install -r requirements.txt
```
or
```
 conda env create -f environment.yaml
```

### Training CLI
🔥🔥**Before training, you need to prepare the dataset in YOLO format in advance and place it in the specified folder, such as '[./datasets](./datasets)'. Modify the dataset YAML file under '[cfg/datas](./cfg/data)'**, and execute the command in the project root directory:
```
python segment/train.py
```
By default, the script will use the DWTAL-s dataset for training and the OG-HFYOLO model for training.

### Valing CLI

```
python segment/val.py
```
🔥🔥**If the mask_nms post-processing mentioned needs to be used, it can be set by adding the parameter '--m_nms'**
```
python segment/val.py --m_nms ...<other param>
```

## License
OG-HFYOLO is released under the GNU Affero General Public License v3.0 (AGPL-3.0). Please see the [LICENSE](./LICENSE) file for more information.

## Copyright Notice
Many utility codes of our project base on the codes of [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
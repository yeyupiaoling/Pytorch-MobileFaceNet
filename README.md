# MobileFaceNet

# 数据集
本项目提供了标注文件，存放在`dataset`目录下，解压即可。另外的图片文件需要自行下载，下载解压到`dataset`目录下即可，如果使用其他数据集，请确保同一个文件夹中的图片是同一个人的，并且是经过对齐裁剪的，大小为`112*112`。
 - emore数据集[百度网盘](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)
 - lfw-align-128下载地址：[百度网盘](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA) 提取码：b2ec

执行下面命令，将提取人脸图片到`dataset/images`。
```shell
python create_dataset.py
```

# 训练

执行`train.py`即可，更多训练参数请查看代码。
```shell
python train.py
```

# 评估

执行`eval.py`即可，更多训练参数请查看代码。
```shell
python eval.py
```

# 预测

在执行预测之前，先要在face_db目录下存放人脸图片，每张图片只包含一个人脸，并以该人脸的名称命名，这建立一个人脸库。之后的识别都会跟这些图片对比，找出匹配成功的人脸。

如果是通过图片路径预测的，请执行下面命令。
```shell
python infer.py --image_path=temp/test.jpg
```

如果是通过相机预测的，请执行下面命令。
```shell
python infer_camera.py --camera_id=0
```
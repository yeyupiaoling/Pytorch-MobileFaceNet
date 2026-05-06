# MobileFaceNet

本项目参考了[ArcFace](https://arxiv.org/abs/1801.07698)的损失函数结合MobileNet，意在开发一个模型较小，但识别准确率较高且推理速度快的一种人脸识别项目，并做了一定的改进, 如模型使用了SE模块，使用了效果更好的AAM损失函数，使用动态调整损失函数的Margin。该项目训练数据使用emore数据集，一共有85742个人，共5822653张图片，使用lfw-align-128数据集作为测试数据。


# 环境
 - Pytorch 2.11.0
 - Python 3.11

# 安装依赖
 1. 安装Pytorch。
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

 2. 安装其他依赖。
```
pip3 install -r requirements.txt
```


# 数据集准备
本项目提供了标注文件，存放在`dataset`目录下，解压即可。另外需要下载下面这两个数据集，下载完解压到`dataset`目录下。
 - emore数据集[百度网盘](https://pan.baidu.com/s/1eXohwNBHbbKXh5KHyItVhQ)
 - lfw-align-128下载地址：[百度网盘](https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA) 提取码：b2ec

然后执行下面命令，制作成训练数据的二进制文件，加快训练时读取速度。
```shell
python create_dataset.py
```

# 训练

执行`train.py`即可，更多训练参数请查看代码。
```shell
python train.py
```

训练输出如下：
```
2026-05-04 11:13:18.832867 Train epoch 0/50, batch: 0/45490, loss: 18.79444, acc: 0.0, lr: 0.00001, eta: 25 days, 14:37:51
2026-05-04 11:13:32.611387 Train epoch 0/50, batch: 100/45490, loss: 18.63379, acc: 0.0, lr: 0.00001, eta: 5 days, 23:32:17
2026-05-04 11:13:46.402430 Train epoch 0/50, batch: 200/45490, loss: 18.69627, acc: 0.0, lr: 0.00001, eta: 5 days, 23:05:42
2026-05-04 11:14:00.193215 Train epoch 0/50, batch: 300/45490, loss: 18.63806, acc: 0.0, lr: 0.00001, eta: 5 days, 23:58:55
2026-05-04 11:14:13.993782 Train epoch 0/50, batch: 400/45490, loss: 18.68702, acc: 0.0, lr: 0.00001, eta: 5 days, 23:56:50
2026-05-04 11:14:27.790523 Train epoch 0/50, batch: 500/45490, loss: 18.91446, acc: 0.0, lr: 0.00001, eta: 5 days, 23:38:44
2026-05-04 11:14:41.593387 Train epoch 0/50, batch: 600/45490, loss: 18.51042, acc: 0.0, lr: 0.00001, eta: 5 days, 23:59:49
2026-05-04 11:14:55.379825 Train epoch 0/50, batch: 700/45490, loss: 18.45938, acc: 0.0, lr: 0.00001, eta: 5 days, 23:52:34
2026-05-04 11:15:09.188305 Train epoch 0/50, batch: 800/45490, loss: 18.62561, acc: 0.0, lr: 0.00001, eta: 5 days, 23:56:10
2026-05-04 11:15:22.993479 Train epoch 0/50, batch: 900/45490, loss: 18.69341, acc: 0.0, lr: 0.00001, eta: 5 days, 23:41:08
```

# 评估

执行`eval.py`即可，更多训练参数请查看代码。
```shell
python eval.py
```

# 预测

本项目已经不教提供了模预测，模型文件可以直接用于预测。在执行预测之前，先要在face_db目录下存放人脸图片，每张图片只包含一个人脸，并以该人脸的名称命名，这建立一个人脸库。之后的识别都会跟这些图片对比，找出匹配成功的人脸。这里使用的人脸检测是MTCNN模型，这个模型具有速度快，模型小的特点，源码地址：[Pytorch-MTCNN](https://github.com/yeyupiaoling/Pytorch-MTCNN)

如果是通过图片路径预测的，请执行下面命令。
```shell
python infer.py --image_path=dataset/test.jpg --face_db_path=face_db --threshold=0.5
```
日志输出如下：
```
人脸检测时间：38ms
人脸识别时间：11ms
人脸对比结果： [('迪丽热巴', 0.7030987), ('杨幂', 0.36442137)]
人脸对比结果： [('杨幂', 0.63616204), ('迪丽热巴', 0.3101096)]
预测的人脸位置： [[272, 67, 328, 118, 1], [156, 80, 215, 134, 1]]
识别的人脸名称： ['迪丽热巴', '杨幂']
总识别时间：82ms
```
![识别结果](./docs/result.jpg)

如果是通过相机预测的，请执行下面命令。
```shell
python infer_camera.py --camera_id=0 --face_db_path=face_db --threshold=0.5
```

# 模型优化记录表

| 优化方式                      | 准确率     |
|---------------------------|---------|
| 上一个版本                     | 0.83017 |
| 优化后                       |         |
| 优化后+SE模块                  |         |
| 优化后+SE模块+margin_scheduler |         |

# C++ 人脸识别推理

该目录提供了一个基于 `LibTorch + OpenCV` 的 C++ 版人脸识别推理实现，包含 MTCNN 人脸检测和 MobileFaceNet 人脸特征提取。

## 目录说明

- `include/mtcnn_detector.h`：MTCNN 检测类声明
- `include/face_recognizer.h`：人脸识别类声明
- `src/mtcnn_detector.cpp`：MTCNN 检测类实现
- `src/face_recognizer.cpp`：人脸识别类实现
- `examples/image_demo.cpp`：图片识别示例
- `examples/camera_demo.cpp`：摄像头实时识别示例
- `export_torchscript_model.py`：模型导出脚本
- `CMakeLists.txt`：CMake 构建文件

## 依赖

- OpenCV
- LibTorch（C++ 版 PyTorch）

OpenCV 下载地址：

 - [Windows OpenCV 4.12.0](https://github.com/opencv/opencv/releases/download/4.12.0/opencv-4.12.0-windows.exe)
 - [Source OpenCV 4.12.0](https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip)

LibTorch 下载地址，如果下载的是GPU版本，要对应自己系统上的CUDA版本：

 - [Windows libtorch 2.11.0（CUDA 13.0）](https://download.pytorch.org/libtorch/cu130/libtorch-win-shared-with-deps-2.11.0%2Bcu130.zip)
 - [Windows libtorch 2.11.0（CPU）](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.11.0%2Bcpu.zip)
 - [Linux libtorch 2.11.0（CUDA 13.0）](https://download.pytorch.org/libtorch/cu130/libtorch-shared-with-deps-2.11.0%2Bcu130.zip)
 - [Linux libtorch 2.11.0（CPU）](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.11.0%2Bcpu.zip)
 - [更多版本下载地址](https://blog.csdn.net/liang_baikai/article/details/127849577)

## 注意事项

> 注意：这里直接加载项目根目录 `save_model` 下的模型文件。MTCNN 模型（`PNet.pth`、`RNet.pth`、`ONet.pth`）和 MobileFaceNet 模型（`mobilefacenet.pth`）需要保证是可被 `torch::jit::load` 加载的 TorchScript 模型。

## 人脸库准备

默认人脸库目录为项目根目录下的 `face_db`，要求：

- 每张图片只包含 1 张人脸
- 文件名作为身份名称
- 支持常见图片格式：`.jpg`、`.jpeg`、`.png`、`.bmp`

例如：

```text
face_db/
├── 杨幂.jpg
└── 迪丽热巴.jpg
```

程序启动时会自动遍历人脸库目录，检测并对齐人脸，提取特征建立内存人脸特征库。

## 构建

以下示例以 Windows 为例，假设：

- OpenCV 已正确安装在 `D:/libs/opencv/build`
- LibTorch 解压路径为 `D:/libs/libtorch`
- 你当前位于项目根目录 `Pytorch-MobileFaceNet/`

1. 进入到cpp目录
```powershell
cd Pytorch-MobileFaceNet/cpp
```

2. 配置CMake并构建
```powershell
cmake -S . -B build -DCMAKE_PREFIX_PATH=D:/libs/libtorch -DOpenCV_DIR=D:/libs/opencv/build
cmake --build build --config Release
```

## 图片识别示例

```powershell
./build/Release/face_image_demo.exe `
  --mtcnn_dir=../save_model/mtcnn `
  --model_path=../save_model/mobilefacenet.pth `
  --face_db=../face_db `
  --image_path=../dataset/test.jpg `
  --threshold=0.6 `
  --device=auto
```

参数说明：

- `--mtcnn_dir`：MTCNN 模型目录，必须包含 `PNet.pth`、`RNet.pth`、`ONet.pth`
- `--model_path`：MobileFaceNet TorchScript 模型路径
- `--face_db`：人脸库目录
- `--image_path`：待识别图片路径
- `--threshold`：相似度阈值，默认 `0.6`
- `--device`：推理设备，可选 `auto`、`cpu`、`cuda`
- `--save_path`：可选，保存可视化结果
- `--show`：是否显示窗口，默认 `1`

带结果保存的示例：

```powershell
./build/Release/face_image_demo.exe `
  --mtcnn_dir=../save_model/mtcnn `
  --model_path=../save_model/mobilefacenet.pth `
  --face_db=../face_db `
  --image_path=../dataset/test.jpg `
  --save_path=result.jpg `
  --threshold=0.6 `
  --device=auto
```

## 摄像头识别示例

```powershell
./build/Release/face_camera_demo.exe `
  --mtcnn_dir=../save_model/mtcnn `
  --model_path=../save_model/mobilefacenet.pth `
  --face_db=../face_db `
  --camera_id=0 `
  --threshold=0.6 `
  --device=auto
```

参数说明：

- `--camera_id`：摄像头编号，默认 `0`

运行后：

- 按 `q` 退出
- 按 `ESC` 退出

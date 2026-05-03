import os
import random
import mmap

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils import data


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('正在加载数据标签...')
        # 获取label
        self.label = {}
        persons_id = set()
        label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, label = line.split(b'\t')
            persons_id.add(int(label))
            self.label[key] = int(label)
        self.num_classes = len(persons_id)
        print('数据加载完成，总数据量为：%d, 类别数量为：%d' % (len(self.label), self.num_classes))

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.label.keys()


# 随机亮度增强
def random_brightness(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(img).enhance(e)


# 随机对比度增强
def random_contrast(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(img).enhance(e)


# 随机颜色强度增强
def random_color(img, lower=0.5, upper=1.5):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Color(img).enhance(e)


# 随机高斯模糊，模拟不同焦距的拍摄
def random_blur(img, p=0.3):
    if random.random() < p:
        radius = random.choice([1, 2])
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


# 随机添加高斯噪声，模拟低光照/传感器噪声
def random_gaussian_noise(img, p=0.3):
    if random.random() < p:
        img_arr = np.array(img, dtype=np.float32)
        sigma = random.uniform(5, 15)
        noise = np.random.normal(0, sigma, img_arr.shape).astype(np.float32)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
    return img


def process(img, image_size=112, is_train=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.resize(img, (image_size, image_size))
    if is_train:
        # 随机水平翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        # 转成PIL进行颜色增强
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color_ops = [random_brightness, random_contrast, random_color]
        np.random.shuffle(color_ops)
        # 依次应用颜色增强，每种50%概率
        for op in color_ops:
            if random.random() > 0.5:
                img = op(img)
        # 随机高斯模糊
        img = random_blur(img, p=0.3)
        # 随机高斯噪声
        img = random_gaussian_noise(img, p=0.3)
        # 转回cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) / 127.5
    # 训练时随机擦除，模拟遮挡
    if is_train and random.random() > 0.5:
        img = random_erasing(img)
    return img


# 随机擦除部分区域，模拟眼镜、口罩等遮挡
def random_erasing(img, sl=0.02, sh=0.2, r1=0.3, r2=3.3):
    c, h, w = img.shape
    area = h * w
    for _ in range(10):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)
        er_h = int(round(np.sqrt(target_area * aspect_ratio)))
        er_w = int(round(np.sqrt(target_area / aspect_ratio)))
        if er_h < h and er_w < w:
            x1 = random.randint(0, h - er_h)
            y1 = random.randint(0, w - er_w)
            # 用随机值填充擦除区域
            img[:, x1:x1 + er_h, y1:y1 + er_w] = random.uniform(-1, 1)
            return img
    return img


class Dataset(data.Dataset):

    def __init__(self, root_path, is_train=True, image_size=112):
        self.imageData = ImageData(root_path)
        self.keys = self.imageData.get_keys()
        self.keys = list(self.keys)
        np.random.shuffle(self.keys)
        self.is_train = is_train
        self.image_size = image_size
        self.num_classes = self.imageData.num_classes

    def __getitem__(self, index):
        key = self.keys[index]
        img = self.imageData.get_img(key)
        assert (img is not None)
        label = self.imageData.get_label(key)
        assert (label is not None)
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = process(img, image_size=self.image_size, is_train=self.is_train)
        img = np.array(img, dtype='float32')
        return img, np.int64(label)

    def __len__(self):
        return len(self.keys)

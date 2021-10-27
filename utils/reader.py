import os
import random
import mmap

import cv2
import numpy as np
from PIL import Image, ImageEnhance
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


def random_brightness(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Brightness(img).enhance(e)


# 随机修改图片的对比度
def random_contrast(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Contrast(img).enhance(e)


# 随机修改图片的颜色强度
def random_color(img, lower=0.7, upper=1.3):
    e = np.random.uniform(lower, upper)
    return ImageEnhance.Color(img).enhance(e)


def process(img, image_size=112, is_train=False):
    if isinstance(img, str):
        img = cv2.imread(img)
    img = cv2.resize(img, (image_size, image_size))
    # 随机水平翻转
    if is_train and random.random() > 0.5:
        img = cv2.flip(img, 1)
    # 图像增强
    if is_train:
        # 转成PIL进行预处理
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ops = [random_brightness, random_contrast, random_color]
        np.random.shuffle(ops)
        if random.random() > 0.5:
            img = ops[0](img)
        # 转回cv2
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) / 127.5
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
        img = np.fromstring(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = process(img, image_size=self.image_size, is_train=self.is_train)
        label = np.array([label], np.int64)
        img = np.array(img, dtype='float32')
        return img, np.array(int(label), dtype=np.int64)

    def __len__(self):
        return len(self.keys)

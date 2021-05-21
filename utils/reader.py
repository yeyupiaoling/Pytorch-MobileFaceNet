import os
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils import data


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


def process(img, image_size=112, is_gray=False, is_train=False):
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
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, np.newaxis]
    img = img.transpose((2, 0, 1))
    img = (img - 127.5) / 127.5
    return img


class Dataset(data.Dataset):

    def __init__(self, root_path, is_train=True, input_shape=(3, 112, 112)):
        self.is_train = is_train
        self.input_shape = input_shape
        self.data = []
        person_id = 0
        persons_dir = os.listdir(root_path)
        for person_dir in persons_dir:
            images = os.listdir(os.path.join(root_path, person_dir))
            for image in images:
                image_path = os.path.join(root_path, person_dir, image)
                self.data.append([image_path, person_id])
            person_id += 1
        random.shuffle(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = process(img_path, is_gray=self.input_shape[0] == 1, is_train=self.is_train)
        img = np.array(img, dtype='float32')
        label = np.int32(label)
        return img, label

    def __len__(self):
        return len(self.data)

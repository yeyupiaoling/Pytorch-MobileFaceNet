import os
import struct
import uuid
from pathlib import Path

import cv2
import mxnet as mx
from tqdm import tqdm


# 从train.rec提前图片到images目录
def load_mx_rec(dataset_path, rec_path):
    save_path = dataset_path / 'images'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()
        path = str(label_path / '{}.jpg'.format(idx))
        cv2.imwrite(path, img)


class DataSetWriter(object):
    def __init__(self, prefix):
        # 创建对应的数据文件
        self.data_file = open(prefix + '.data', 'wb')
        self.header_file = open(prefix + '.header', 'wb')
        self.label_file = open(prefix + '.label', 'wb')
        self.offset = 0
        self.header = ''

    def add_img(self, key, img):
        # 写入图像数据
        self.data_file.write(struct.pack('I', len(key)))
        self.data_file.write(key.encode('ascii'))
        self.data_file.write(struct.pack('I', len(img)))
        self.data_file.write(img)
        self.offset += 4 + len(key) + 4
        self.header = key + '\t' + str(self.offset) + '\t' + str(len(img)) + '\n'
        self.header_file.write(self.header.encode('ascii'))
        self.offset += len(img)

    def add_label(self, label):
        # 写入标签数据
        self.label_file.write(label.encode('ascii') + '\n'.encode('ascii'))


# 人脸识别训练数据的格式转换
def convert_data(root_path, output_prefix):
    # 读取全部的数据类别获取数据
    person_id = 0
    data = []
    persons_dir = os.listdir(root_path)
    for person_dir in persons_dir:
        images = os.listdir(os.path.join(root_path, person_dir))
        for image in images:
            image_path = os.path.join(root_path, person_dir, image)
            data.append([image_path, person_id])
        person_id += 1
    print("训练数据大小：%d，总类别为：%d" % (len(data), person_id))

    # 开始写入数据
    writer = DataSetWriter(output_prefix)
    for image_path, person_id in tqdm(data):
        try:
            key = str(uuid.uuid1())
            img = cv2.imread(image_path)
            _, img = cv2.imencode('.bmp', img)
            # 写入对应的数据
            writer.add_img(key, img.tostring())
            label_str = str(person_id)
            writer.add_label('\t'.join([key, label_str]))
        except:
            continue


if __name__ == '__main__':
    load_mx_rec(Path('dataset'), Path('dataset/faces_emore'))
    convert_data('dataset/images', 'dataset/train_data')

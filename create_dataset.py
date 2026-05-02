import os
import shutil
import struct
import uuid
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# 兼容 MXNet RecordIO 的头结构
IRHeader = namedtuple('HEADER', ['flag', 'label', 'id', 'id2'])
_IR_FORMAT = 'IfQQ'
_IR_SIZE = struct.calcsize(_IR_FORMAT)
_REC_MAGIC = 0xced7230a


def _load_recordio_index(idx_path):
    """读取 train.idx，返回按位置排序后的索引信息。"""
    index_items = []
    with open(idx_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, pos = line.split('\t')
            index_items.append((int(key), int(pos)))
    index_items.sort(key=lambda item: item[1])
    return index_items


def _extract_record_bytes(record_chunk):
    """从一段 RecordIO 原始字节中恢复出完整的单条记录内容。"""
    pieces = []
    offset = 0
    while offset < len(record_chunk):
        if offset + 8 > len(record_chunk):
            raise ValueError('RecordIO 记录头不完整')
        magic, lrecord = struct.unpack_from('<II', record_chunk, offset)
        if magic != _REC_MAGIC:
            raise ValueError('RecordIO 魔数不匹配')

        # 高 3 位是连续标记，低 29 位是当前分片长度
        cflag = (lrecord >> 29) & 0x7
        length = lrecord & ((1 << 29) - 1)
        offset += 8

        if offset + length > len(record_chunk):
            raise ValueError('RecordIO 分片长度越界')
        pieces.append(record_chunk[offset:offset + length])
        offset += length

        # RecordIO 会按 4 字节对齐，因此每个分片末尾可能带有 padding
        offset += (- (8 + length)) % 4

        if cflag == 0 or cflag == 3:
            break
        if cflag not in (1, 2):
            raise ValueError('RecordIO 连续标记非法: {}'.format(cflag))

    return b''.join(pieces)


def _unpack_record(record_bytes):
    """按 MXNet 的打包格式解析记录头和图像字节。"""
    if len(record_bytes) < _IR_SIZE:
        raise ValueError('RecordIO 数据长度不足')

    header = IRHeader(*struct.unpack(_IR_FORMAT, record_bytes[:_IR_SIZE]))
    payload = record_bytes[_IR_SIZE:]
    if header.flag > 0:
        label = np.frombuffer(payload[:header.flag * 4], dtype=np.float32)
        header = header._replace(label=label)
        payload = payload[header.flag * 4:]
    return header, payload


def _get_scalar_label(label):
    """统一处理标量标签和数组标签。"""
    if isinstance(label, np.ndarray):
        return int(label[0])
    return int(label)


# 直接从 train.rec 读取图片，并写成 DataSetWriter 的二进制格式
def load_mx_rec(dataset_path, rec_path):
    if not rec_path.exists():
        raise FileNotFoundError(f'记录文件路径不存在: {rec_path}，请检查是否正确下载并解压到dataset目录下')
    idx_path = rec_path / 'train.idx'
    rec_file_path = rec_path / 'train.rec'
    output_prefix = str(dataset_path / 'train_data')

    index_items = _load_recordio_index(idx_path)
    positions = {key: pos for key, pos in index_items}
    sorted_keys = [key for key, _ in index_items]
    next_positions = {}
    for i, key in enumerate(sorted_keys):
        next_positions[key] = index_items[i + 1][1] if i + 1 < len(index_items) else None

    if 0 not in positions:
        raise ValueError('train.idx 中缺少索引 0，无法读取数据集元信息')

    label_map = {}
    writer = DataSetWriter(output_prefix)
    try:
        with open(rec_file_path, 'rb') as rec_file:
            rec_file.seek(0, os.SEEK_END)
            rec_file_size = rec_file.tell()

            # 第 0 条记录保存的是数据集元信息，其中 label[0] 是最大索引
            zero_pos = positions[0]
            zero_end = next_positions[0] if next_positions[0] is not None else rec_file_size
            rec_file.seek(zero_pos)
            dataset_info = rec_file.read(zero_end - zero_pos)
            header, _ = _unpack_record(_extract_record_bytes(dataset_info))
            max_idx = _get_scalar_label(header.label)

            for idx in tqdm(range(1, max_idx), desc='转换 train.rec'):
                if idx not in positions:
                    continue

                current_pos = positions[idx]
                next_pos = next_positions[idx] if next_positions[idx] is not None else rec_file_size
                rec_file.seek(current_pos)
                record_chunk = rec_file.read(next_pos - current_pos)

                try:
                    header, img_bytes = _unpack_record(_extract_record_bytes(record_chunk))
                    raw_label = _get_scalar_label(header.label)
                    if raw_label not in label_map:
                        label_map[raw_label] = len(label_map)
                    person_id = label_map[raw_label]

                    # 直接复用 rec 中已编码好的图片字节，避免重复解码/编码
                    key = str(uuid.uuid1())
                    writer.add_img(key, img_bytes)
                    writer.add_label('\t'.join([key, str(person_id)]))
                except Exception:
                    continue
        
        # 删除旧文件夹
        shutil.rmtree('dataset/faces_emore', ignore_errors=True)
    finally:
        writer.close()
    print('训练数据转换完成，总类别为：%d' % len(label_map))


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

    def close(self):
        self.data_file.close()
        self.header_file.close()
        self.label_file.close()


if __name__ == '__main__':
    load_mx_rec(Path('dataset'), Path('dataset/faces_emore'))

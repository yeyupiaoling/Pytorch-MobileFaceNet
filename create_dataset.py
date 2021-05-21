import cv2
from pathlib import Path
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


if __name__ == '__main__':
    load_mx_rec(Path('dataset'), Path('dataset/faces_emore'))
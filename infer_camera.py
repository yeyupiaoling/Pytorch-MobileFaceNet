import argparse
import functools
import time

import cv2
from utils.predictor import Predictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('camera_id',                int,     0,                                  '使用的相机ID')
add_arg('face_db_path',             str,     'face_db',                          '人脸库路径')
add_arg('threshold',                float,   0.6,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)


def main():
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path, threshold=args.threshold)
    cap = cv2.VideoCapture(args.camera_id)
    while True:
        ret, img = cap.read()
        if ret:
            start = time.time()
            results = predictor.recognition(img)
            if results is not None:
                img = predictor.draw_face(img, results)
                print('识别结果：', results)
                print(f'总识别时间：{int((time.time() - start) * 1000)}ms')
            cv2.imshow("result", img)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()

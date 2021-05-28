import os
import sys
import cv2
import numpy as np
import torch
from skimage import transform as trans

from .utils import generate_bbox, py_nms, convert_to_square
from .utils import pad, calibrate_box, processed_image


class MTCNN:
    def __init__(self, model_path):
        self.device = torch.device("cuda")
        # 获取P模型
        self.pnet = torch.jit.load(os.path.join(model_path, 'PNet.pth'))
        self.pnet.to(self.device)
        self.softmax_p = torch.nn.Softmax(dim=0)
        self.pnet.eval()

        # 获取R模型
        self.rnet = torch.jit.load(os.path.join(model_path, 'RNet.pth'))
        self.rnet.to(self.device)
        self.softmax_r = torch.nn.Softmax(dim=-1)
        self.rnet.eval()

        # 获取R模型
        self.onet = torch.jit.load(os.path.join(model_path, 'ONet.pth'))
        self.onet.to(self.device)
        self.softmax_o = torch.nn.Softmax(dim=-1)
        self.onet.eval()

    # 使用PNet模型预测
    def predict_pnet(self,infer_data):
        # 添加待预测的图片
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        infer_data = torch.unsqueeze(infer_data, dim=0)
        # 执行预测
        cls_prob, bbox_pred, _ = self.pnet(infer_data)
        cls_prob = torch.squeeze(cls_prob)
        cls_prob = self.softmax_p(cls_prob)
        bbox_pred = torch.squeeze(bbox_pred)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()

    # 使用RNet模型预测
    def predict_rnet(self,infer_data):
        # 添加待预测的图片
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        # 执行预测
        cls_prob, bbox_pred, _ = self.rnet(infer_data)
        cls_prob = self.softmax_r(cls_prob)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy()

    # 使用ONet模型预测
    def predict_onet(self,infer_data):
        # 添加待预测的图片
        infer_data = torch.tensor(infer_data, dtype=torch.float32, device=self.device)
        # 执行预测
        cls_prob, bbox_pred, landmark_pred = self.onet(infer_data)
        cls_prob = self.softmax_o(cls_prob)
        return cls_prob.detach().cpu().numpy(), bbox_pred.detach().cpu().numpy(), landmark_pred.detach().cpu().numpy()

    # 获取PNet网络输出结果
    def detect_pnet(self,im, min_face_size, scale_factor, thresh):
        """通过pnet筛选box和landmark
        参数：
          im:输入图像[h,2,3]
        """
        net_size = 12
        # 人脸和输入图像的比率
        current_scale = float(net_size) / min_face_size
        im_resized = processed_image(im, current_scale)
        _, current_height, current_width = im_resized.shape
        all_boxes = list()
        # 图像金字塔
        while min(current_height, current_width) > net_size:
            # 类别和box
            cls_cls_map, reg = self.predict_pnet(im_resized)
            boxes = generate_bbox(cls_cls_map[1, :, :], reg, current_scale, thresh)
            current_scale *= scale_factor  # 继续缩小图像做金字塔
            im_resized = processed_image(im, current_scale)
            _, current_height, current_width = im_resized.shape

            if boxes.size == 0:
                continue
            # 非极大值抑制留下重复低的box
            keep = py_nms(boxes[:, :5], 0.5, mode='Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)
        if len(all_boxes) == 0:
            return None
        all_boxes = np.vstack(all_boxes)
        # 将金字塔之后的box也进行非极大值抑制
        keep = py_nms(all_boxes[:, 0:5], 0.7, mode='Union')
        all_boxes = all_boxes[keep]
        # box的长宽
        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1
        # 对应原图的box坐标和分数
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes_c

    # 获取RNet网络输出结果
    def detect_rnet(self,im, dets, thresh):
        """通过rent选择box
            参数：
              im：输入图像
              dets:pnet选择的box，是相对原图的绝对坐标
            返回值：
              box绝对坐标
        """
        h, w, c = im.shape
        # 将pnet的box变成包含它的正方形，可以避免信息损失
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        # 调整超出图像的box
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        delete_size = np.ones_like(tmpw) * 20
        ones = np.ones_like(tmpw)
        zeros = np.zeros_like(tmpw)
        num_boxes = np.sum(np.where((np.minimum(tmpw, tmph) >= delete_size), ones, zeros))
        cropped_ims = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        for i in range(int(num_boxes)):
            # 将pnet生成的box相对与原图进行裁剪，超出部分用0补
            if tmph[i] < 20 or tmpw[i] < 20:
                continue
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            try:
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
                img = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_LINEAR)
                img = img.transpose((2, 0, 1))
                img = (img - 127.5) / 128
                cropped_ims[i, :, :, :] = img
            except:
                continue
        cls_scores, reg = self.predict_rnet(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
        else:
            return None

        keep = py_nms(boxes, 0.4, mode='Union')
        boxes = boxes[keep]
        # 对pnet截取的图像的坐标进行校准，生成rnet的人脸框对于原图的绝对坐标
        boxes_c = calibrate_box(boxes, reg[keep])
        return boxes_c

    # 获取ONet模型预测结果
    def detect_onet(self,im, dets, thresh):
        """将onet的选框继续筛选基本和rnet差不多但多返回了landmark"""
        h, w, c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]
        cropped_ims = np.zeros((num_boxes, 3, 48, 48), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            img = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_LINEAR)
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 128
            cropped_ims[i, :, :, :] = img
        cls_scores, reg, landmark = self.predict_onet(cropped_ims)

        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > thresh)[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        w = boxes[:, 2] - boxes[:, 0] + 1

        h = boxes[:, 3] - boxes[:, 1] + 1
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        boxes_c = calibrate_box(boxes, reg)

        keep = py_nms(boxes_c, 0.6, mode='Minimum')
        boxes_c = boxes_c[keep]
        landmark = landmark[keep]
        return boxes_c, landmark

    def infer_image_path(self, image_path):
        im = cv2.imread(image_path)
        # 调用第一个模型预测
        boxes_c = self.detect_pnet(im, 20, 0.79, 0.9)
        if boxes_c is None:
            return None, None
        # 调用第二个模型预测
        boxes_c = self.detect_rnet(im, boxes_c, 0.6)
        if boxes_c is None:
            return None, None
        # 调用第三个模型预测
        boxes_c, landmark = self.detect_onet(im, boxes_c, 0.7)
        if boxes_c is None:
            return None, None

        return boxes_c, landmark

    # 对齐
    @staticmethod
    def estimate_norm(lmk):
        assert lmk.shape == (5, 2)
        tform = trans.SimilarityTransform()
        src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
        tform.estimate(lmk, src)
        M = tform.params[0:2, :]
        return M

    def norm_crop(self, img, landmark, image_size=112):
        M = self.estimate_norm(landmark)
        warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
        return warped

    def infer_image(self, im):
        if isinstance(im, str):
            im = cv2.imread(im)
        # 调用第一个模型预测
        boxes_c = self.detect_pnet(im, 20, 0.79, 0.9)
        if boxes_c is None:
            return None, None
        # 调用第二个模型预测
        boxes_c = self.detect_rnet(im, boxes_c, 0.6)
        if boxes_c is None:
            return None, None
        # 调用第三个模型预测
        boxes_c, landmarks = self.detect_onet(im, boxes_c, 0.7)
        if boxes_c is None:
            return None, None
        imgs = []
        for landmark in landmarks:
            landmark = [[float(landmark[i]), float(landmark[i + 1])] for i in range(0, len(landmark), 2)]
            landmark = np.array(landmark, dtype='float32')
            img = self.norm_crop(im, landmark)
            imgs.append(img)

        return imgs, boxes_c

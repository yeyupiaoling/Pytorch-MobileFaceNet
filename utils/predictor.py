import os

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN


class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7, image_size=112):
        self.threshold = threshold
        self.image_size = image_size
        self.face_db_path = face_db_path
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        self.device = torch.device("cuda")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path)
        self.model.to(self.device)
        self.model.eval()
        logger.info('模型加载完成')

        self.update_face_db()

    def update_face_db(self):
        self.faces_db = self.load_face_db(self.face_db_path)
        logger.info('人脸库更新完成')
    
    # 批量加载人脸库 - 优化批量处理
    def load_face_db(self, face_db_path):
        # 第一步：批量收集所有图片和对应的名称
        valid_imgs = []
        valid_names = []
        
        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            
            # 人脸检测
            imgs, _ = self.mtcnn.infer_image(img)
            
            # 只处理恰好一张人脸的图片
            if imgs is None or len(imgs) > 1:
                logger.warning(f'人脸库中的 {image_path} 图片包含不是1张人脸，自动跳过该图片')
                continue
            
            # 预处理图片
            processed_imgs = self.process(imgs)
            valid_imgs.extend(processed_imgs)
            valid_names.append(name)
        
        # 第二步：批量特征提取 - 使用优化后的 infer 方法一次性处理所有图片
        if len(valid_imgs) == 0:
            logger.warning('人脸库中没有有效的人脸图片')
            return {}
        
        # 转换为批量格式
        imgs_batch = np.array(valid_imgs, dtype='float32')
        features_batch = self.infer(imgs_batch)
        
        # 构建人脸库字典
        faces_db = {}
        for name, feature in zip(valid_names, features_batch):
            faces_db[name] = feature[0]
        
        return faces_db

    def process(self, imgs):
        new_imgs = []
        for img in imgs:    
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            new_imgs.append(img)
        return new_imgs

    # 预测图片 - 优化批量推理
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        
        # 一次性转换所有图片为tensor，避免循环中的重复转换
        imgs_tensor = torch.tensor(imgs, dtype=torch.float32, device=self.device)
        
        # 批量推理 - 一次性处理所有图片
        features = self.model(imgs_tensor)
        features = features.detach().cpu().numpy()
        
        # 返回列表格式，每个元素对应一张图片的特征
        return [features[i:i+1] for i in range(features.shape[0])]

    # 人脸识别 - 优化批量计算相似度
    def recognition(self, img):
        imgs, boxes = self.mtcnn.infer_image(img)
        if imgs is None:
            return None, None
        
        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        
        # 批量提取特征 - 使用优化后的 infer 方法一次性处理所有图片
        features = self.infer(imgs)
        
        # 将所有特征拼接成矩阵，用于批量计算相似度
        features_matrix = np.array([f[0] for f in features])
        
        # 批量计算与所有人脸库特征的相似度
        # faces_db_matrix: (num_db, feature_dim)
        faces_db_names = list(self.faces_db.keys())
        faces_db_matrix = np.array([self.faces_db[name] for name in faces_db_names])
        
        # 计算相似度矩阵：(num_faces, num_db)
        # 使用公式：cosine_similarity = dot(a, b) / (norm(a) * norm(b))
        features_norm = features_matrix / (np.linalg.norm(features_matrix, axis=1, keepdims=True) + 1e-8)
        faces_db_norm = faces_db_matrix / (np.linalg.norm(faces_db_matrix, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(features_norm, faces_db_norm.T)
        
        # 获取每个脸的最高相似度及其索引
        max_indices = np.argmax(similarity_matrix, axis=1)
        max_probs = similarity_matrix[np.arange(len(features)), max_indices]
        
        # 构建结果
        results = []
        for idx, prob in zip(max_indices, max_probs):
            prob_float = round(prob.item(), 4)
            bbox = boxes[idx].astype(np.int64).tolist()[:4]
            name = faces_db_names[idx]
            logger.info(f'人脸对比结果：{name} - 相似度: {prob_float:.4f}')
            
            # 判断是否超过阈值
            if prob_float > self.threshold:
                results.append({"name": name, "prob": prob_float, "bbox": bbox})
            else:
                results.append({"name": "unknow", "prob": prob_float, "bbox": bbox})
        
        return results

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font_path = os.path.join(os.path.dirname(__file__), 'simfang.ttf')
        font = ImageFont.truetype(font_path, size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, img, results):
        if isinstance(img, str):
            img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
        if results is not None:
            for result in results:
                bbox = result["bbox"]
                name = result["name"]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]), (corpbbox[2], corpbbox[3]), (0, 0, 255), 2)
                # 判别为人脸的名字
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] - 20, color=(255, 0, 0), size=16)
        return img

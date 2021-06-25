import distutils.util
from tqdm import tqdm
import cv2
import numpy as np
import torch
from sklearn import preprocessing


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' 默认: %(default)s.',
                           **kwargs)


# 获取lfw全部路径
def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


# 加载图片并预处理
def load_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.resize(image, (112, 112))
    image_flip = np.fliplr(image)
    image = np.array([image, image_flip], dtype='float32')
    image = image.transpose((0, 3, 1, 2))
    image = image.astype(np.float32, copy=False)
    image = (image - 127.5) / 127.5
    return image


# 获取图像特征
def get_features(model, test_list, batch_size=32):
    images = None
    features = None
    for i, img_path in enumerate(tqdm(test_list)):
        image = load_image(img_path)
        assert image is not None, '{} 图片错误'.format(img_path)

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()
            output = preprocessing.normalize(output)

            feature_1 = output[0::2]
            feature_2 = output[1::2]
            feature = np.hstack((feature_1, feature_2))

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))
            images = None
    return features


# 加载模型文件
def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# 将文件路径名跟模型输出的图像特征打包成字典
def get_feature_dict(test_list, features):
    feature_dict = {}
    for i, each in enumerate(test_list):
        feature_dict[each] = features[i]
    return feature_dict


# 计算对角余弦值
def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 根据对角余弦值计算准确率
def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_accuracy = 0
    best_threshold = 0
    for i in range(len(y_score)):
        threshold = y_score[i]
        y_test = (y_score >= threshold)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return best_accuracy, best_threshold


# 计算lfw每一对的相似度
def test_performance(feature_dict, lfw_data_list):
    with open(lfw_data_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        feature_1 = feature_dict[splits[0]]
        feature_2 = feature_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(feature_1, feature_2)

        sims.append(sim)
        labels.append(label)

    accuracy, threshold = cal_accuracy(sims, labels)
    return accuracy, threshold

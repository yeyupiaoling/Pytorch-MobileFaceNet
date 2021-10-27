import argparse
import functools

import torch

from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',       int,    64,                                 '训练的批量大小')
add_arg('test_list_path',   str,    'dataset/lfw_test.txt',             '测试数据的数据列表路径')
add_arg('model_path',       str,    'save_model/mobilefacenet.pth',     '模型保存的路径')
args = parser.parse_args()


def eval(args, model):
    # 获取测试数据
    img_paths = get_lfw_list(args.test_list_path)
    # 开始预测
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    accuracy, threshold = test_performance(fe_dict, args.test_list_path)
    return accuracy, threshold


def main():
    # 获取模型
    model = torch.load(args.model_path)
    model.to(torch.device("cuda"))

    model.eval()
    accuracy, threshold = eval(args, model)
    print('准确率为：%f, 最优阈值为：%f' % (accuracy, threshold))


if __name__ == '__main__':
    print_arguments(args)
    main()

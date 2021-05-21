import os
from datetime import datetime
import argparse
import functools
import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.reader import Dataset
from models.metrics import ArcMarginNet
from models.resnet import resnet_face34
from models.mobilefacenet import MobileFaceNet
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpu',              str,    '0,1',                    '训练使用的GPU序号')
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    2,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    100,                      '训练的轮数')
add_arg('num_classes',      int,    85742,                    '分类的类别数量')
add_arg('learning_rate',    float,  1e-1,                     '初始学习率的大小')
add_arg('weight_decay',     float,  5e-4,                     'weight_decay的大小')
add_arg('lr_step',          int,    10,                       '学习率衰减步数')
add_arg('use_model',        str,    'mobilefacenet',          '所使用的模型，支持mobilefacenet，resnet_face34')
add_arg('optimizer',        str,    'sgd',                    '所使用的优化方法')
add_arg('train_root_path',  str,    'dataset/images',         '训练数据的根目录')
add_arg('test_list_path',   str,    'dataset/lfw_test.txt',   '测试数据的数据列表路径')
add_arg('save_model',       str,    'save_model/',            '模型保存的路径')
args = parser.parse_args()


@torch.no_grad()
def test(args, model):
    # 获取测试数据
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    accuracy, _ = test_performance(fe_dict, args.test_list_path)
    return accuracy


def save_model(model, save_path, epoch_id):
    if os.path.exists(os.path.join(save_path, '%s_%d.pth' % (args.use_model, epoch_id - 3))):
        os.remove(os.path.join(save_path, '%s_%d.pth' % (args.use_model, epoch_id - 3)))
    save_path = os.path.join(save_path, '%s_%d.pth' % (args.use_model, epoch_id))
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(model, save_path)


def train():
    device = torch.device("cuda")
    # 获取模型
    if args.use_model == 'mobilefacenet':
        model = MobileFaceNet()
    else:
        model = resnet_face34(use_se=False)
    metric_fc = ArcMarginNet(512, args.num_classes, s=64, m=0.5)

    model.to(device)
    summary(model, (3, 112, 112))
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    # 获取优化方法
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=args.lr, weight_decay=args.weight_decay)
    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)

    # 获取损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 获取训练数据
    train_dataset = Dataset(args.train_root_path, is_train=True, input_shape=(3, 112, 112))
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    # 开始训练
    for epoch_id in range(args.num_epoch):
        model.train()
        for batch_id, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)

            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 100 == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                print('[{}] Train epoch {} Batch {} Loss {:.5} Accuracy {}'.format(datetime.now(), epoch_id, batch_id, loss.item(), acc))
        scheduler.step()
        # 开始评估
        model.eval()
        accuracy = test(args, model)
        print('[{}] Test epoch {} Accuracy {:.5}'.format(datetime.now(), epoch_id, accuracy))

        # 保存模型
        save_model(model, args.save_model, epoch_id)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print_arguments(args)
    train()

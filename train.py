import os
import re
import shutil
import time
from datetime import datetime, timedelta
import argparse
import functools
import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.reader import Dataset
from models.arcmargin import ArcNet
from models.mobilefacenet import MobileFaceNet
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('gpus',             str,    '0',                      '训练使用的GPU序号')
add_arg('batch_size',       int,    64,                       '训练的批量大小')
add_arg('num_workers',      int,    0,                        '读取数据的线程数量')
add_arg('num_epoch',        int,    50,                       '训练的轮数')
add_arg('learning_rate',    float,  1e-3,                     '初始学习率的大小')
add_arg('train_root_path',  str,    'dataset/train_data',     '训练数据的根目录')
add_arg('test_list_path',   str,    'dataset/lfw_test.txt',   '测试数据的数据列表路径')
add_arg('save_model',       str,    'save_model/',            '模型保存的路径')
add_arg('resume',           str,    None,                     '恢复训练，当为None则不使用恢复模型')
args = parser.parse_args()


@torch.no_grad()
def test(args, model):
    # 获取测试数据
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    accuracy, _ = test_performance(fe_dict, args.test_list_path)
    return accuracy


def save_model(args, model, metric_fc, optimizer, epoch_id):
    model_params_path = os.path.join(args.save_model, 'epoch_%d' % epoch_id)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数和优化方法参数
    torch.save(model.state_dict(), os.path.join(model_params_path, 'model_params.pth'))
    torch.save(metric_fc.state_dict(), os.path.join(model_params_path, 'metric_fc_params.pth'))
    torch.save(optimizer.state_dict(), os.path.join(model_params_path, 'optimizer.pth'))
    # 删除旧的模型
    old_model_path = os.path.join(args.save_model, 'epoch_%d' % (epoch_id - 3))
    if os.path.exists(old_model_path):
        shutil.rmtree(old_model_path)
    # 保存整个模型和参数
    all_model_path = os.path.join(args.save_model, 'mobilefacenet.pth')
    if not os.path.exists(os.path.dirname(all_model_path)):
        os.makedirs(os.path.dirname(all_model_path))
    torch.jit.save(torch.jit.script(model), all_model_path)


def train():
    device_ids = [int(i) for i in args.gpus.split(',')]
    # 获取训练数据
    train_dataset = Dataset(args.train_root_path, is_train=True, image_size=112)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size * len(device_ids),
                              shuffle=True,
                              num_workers=args.num_workers)
    print("[%s] 总数据类别为：%d" % (datetime.now(), train_dataset.num_classes))

    device = torch.device("cuda")
    # 获取模型
    model = MobileFaceNet()
    metric_fc = ArcNet(512, train_dataset.num_classes)
    if len(device_ids) > 1:
        model = DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
        metric_fc = DataParallel(metric_fc, device_ids=device_ids, output_device=device_ids[0])

    model.to(device)
    metric_fc.to(device)
    if len(args.gpus.split(',')) > 1:
        summary(model.module, (3, 112, 112))
    else:
        summary(model, (3, 112, 112))

    # 初始化epoch数
    last_epoch = 0
    # 获取优化方法
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    # 获取学习率衰减函数
    scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

    # 获取损失函数
    criterion = torch.nn.CrossEntropyLoss()

    # 加载模型参数和优化方法参数
    if args.resume:
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = int(re.findall("\d+", args.resume)[-1]) + 1
        if len(device_ids) > 1:
            model.module.load_state_dict(torch.load(os.path.join(args.resume, 'model_params.pth')))
            metric_fc.module.load_state_dict(torch.load(os.path.join(args.resume, 'metric_fc_params.pth')))
        else:
            model.load_state_dict(torch.load(os.path.join(args.resume, 'model_params.pth')))
            metric_fc.load_state_dict(torch.load(os.path.join(args.resume, 'metric_fc_params.pth')))
        print('成功加载模型参数和优化方法参数')

    # 开始训练
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    for epoch_id in range(last_epoch, args.num_epoch):
        start = time.time()
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
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print('[%s] Train epoch %d, batch: %d/%d, loss: %f, accuracy: %f, lr: %f, eta: %s' % (
                    datetime.now(), epoch_id, batch_id, len(train_loader), loss.item(), acc.item(), scheduler.get_lr()[0], eta_str))
            start = time.time()
        scheduler.step()
        # 开始评估
        model.eval()
        print('='*70)
        accuracy = test(args, model)
        model.train()
        print('[{}] Test epoch {} Accuracy {:.5}'.format(datetime.now(), epoch_id, accuracy))
        print('='*70)

        # 保存模型
        if len(args.gpus.split(',')) > 1:
            save_model(args, model.module, metric_fc.module, optimizer, epoch_id)
        else:
            save_model(args, model, metric_fc, optimizer, epoch_id)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    print_arguments(args)
    train()

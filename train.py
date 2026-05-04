import os
import re
import shutil
import time
from datetime import datetime, timedelta
import argparse
import functools
import numpy as np
from visualdl import LogWriter
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from utils.reader import Dataset
from models.aamloss import AAMLoss
from models.fc import Classifier
from models.mobilefacenet import MobileFaceNet
from utils.scheduler import MarginScheduler, WarmupCosineSchedulerLR
from utils.utils import add_arguments, print_arguments, get_lfw_list
from utils.utils import get_features, get_feature_dict, test_performance


parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',           int,    128,                      '训练的批量大小')
add_arg('num_workers',          int,    4,                        '读取数据的线程数量')
add_arg('num_epoch',            int,    50,                       '训练的轮数')
add_arg('min_lr',               float,  1e-5,                     '最小学习率')
add_arg('max_lr',               float,  1e-3,                     '最大学习率')
add_arg('warmup_epoch',         int,    5,                        '学习率预热的轮数')
add_arg('image_size',           int,    112,                      '训练输入到模型的图像大小')
add_arg('train_root_path',      str,    'dataset/train_data',     '训练数据的根目录')
add_arg('test_list_path',       str,    'dataset/lfw_test.txt',   '测试数据的数据列表路径')
add_arg('save_model',           str,    'save_model/',            '模型保存的路径')
add_arg('resume',               str,    None,                     '恢复训练，当为None则不使用恢复模型')
add_arg('use_se',               bool,   True,                     '是否使用SE模块')
add_arg('use_margin_scheduler', bool,   False,                    '是否使用动态调整损失函数Margin')
args = parser.parse_args()


@torch.no_grad()
def test(args, model):
    # 获取测试数据
    img_paths = get_lfw_list(args.test_list_path)
    features = get_features(model, img_paths, batch_size=args.batch_size)
    fe_dict = get_feature_dict(img_paths, features)
    accuracy, _ = test_performance(fe_dict, args.test_list_path)
    return accuracy


def save_model(args, model, classifier, optimizer, epoch_id):
    model_params_path = os.path.join(args.save_model, 'epoch_%d' % epoch_id)
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
    # 保存模型参数和优化方法参数
    torch.save(model.state_dict(), os.path.join(model_params_path, 'model_params.pth'))
    torch.save(classifier.state_dict(), os.path.join(model_params_path, 'classifier_params.pth'))
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
    # 如果是Windows，args.num_workers必须为0
    if os.name == 'nt':
        args.num_workers = 0
    # 获取训练数据
    train_dataset = Dataset(args.train_root_path, is_train=True, image_size=args.image_size)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    print("[%s] 总数据类别为：%d" % (datetime.now(), train_dataset.num_classes))

    device = torch.device("cuda")
    # 获取模型
    model = MobileFaceNet(use_se=args.use_se)
    # 分类层：特征归一化+分类权重归一化，输出余弦相似度logits
    classifier = Classifier(512, train_dataset.num_classes)
    # AAM损失函数：在角度空间上增加margin提升判别性
    aam_loss = AAMLoss(margin=0.2, scale=32.0)
    model.to(device)
    classifier.to(device)
    summary(model, (3, 112, 112))

    # 初始化epoch数
    last_epoch = 0
    # 获取优化方法，增大weight_decay防止过拟合
    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
    # 学习率预热+余弦退火调度（step级别，每个batch更新一次）
    scheduler = WarmupCosineSchedulerLR(optimizer, fix_epoch=args.num_epoch, step_per_epoch=len(train_loader),
                                        min_lr=args.min_lr, max_lr=args.max_lr, warmup_epoch=args.warmup_epoch)
    margin_scheduler = None
    if args.use_margin_scheduler:
        margin_scheduler = MarginScheduler(criterion=aam_loss, step_per_epoch=len(train_loader),
                                           increase_start_epoch=int(args.num_epoch * 0.3),
                                           fix_epoch=int(args.num_epoch * 0.7))
    # 加载模型参数和优化方法参数
    if args.resume:
        optimizer_state = torch.load(os.path.join(args.resume, 'optimizer.pth'))
        optimizer.load_state_dict(optimizer_state)
        # 获取预训练的epoch数
        last_epoch = int(args.resume.split('/')[-1].split('_')[-1]) + 1
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model_params.pth')))
        classifier.load_state_dict(torch.load(os.path.join(args.resume, 'classifier_params.pth')))
        print('成功加载模型参数和优化方法参数')

    # 日志记录器
    writer = LogWriter(logdir='log/')
    train_step = 0
    # 开始训练
    sum_batch = len(train_loader) * (args.num_epoch - last_epoch)
    for epoch_id in range(last_epoch, args.num_epoch):
        start = time.time()
        for batch_id, data in enumerate(train_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            # Classifier内部完成归一化和logits计算，直接传给AAMLoss
            logits = classifier(feature)
            loss = aam_loss(logits, label)
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            if margin_scheduler is not None:
                margin_scheduler.step()

            if batch_id % 100 == 0:
                # 计算训练准确率（使用AAM缩放后的输出）
                aam_output = logits * aam_loss.scale
                output_pred = aam_output.data.cpu().numpy()
                output_pred = np.argmax(output_pred, axis=1)
                label_np = label.data.cpu().numpy()
                acc = np.mean((output_pred == label_np).astype(int))
                eta_sec = ((time.time() - start) * 1000) * (sum_batch - (epoch_id - last_epoch) * len(train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                print(f'{datetime.now()} Train epoch {epoch_id}/{args.num_epoch}, batch: {batch_id}/{len(train_loader)}, loss: {loss.item():.5f}, '
                      f'acc: {acc.item()}, lr: {scheduler.get_last_lr()[0]:.5f}, eta: {eta_str}')
                # 记录训练损失和学习率
                writer.add_scalar('Train/Loss', loss.item(), train_step)
                writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], train_step)
                if margin_scheduler is not None:
                    writer.add_scalar('Train/margin', margin_scheduler.get_margin(), train_step)
                train_step += 1
            start = time.time()
        # 开始评估
        model.eval()
        print('='*70)
        accuracy = test(args, model)
        model.train()
        print(f'{datetime.now()} Test epoch {epoch_id}/{args.num_epoch} Accuracy {accuracy:.5f}')
        # 记录测试准确率
        writer.add_scalar('Test/Accuracy', accuracy, epoch_id)
        print('='*70)

        # 保存模型
        save_model(args, model, classifier, optimizer, epoch_id)


if __name__ == '__main__':
    print_arguments(args)
    train()

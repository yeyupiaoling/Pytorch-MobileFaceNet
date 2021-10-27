import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, Linear
from torch.nn import Conv2d, BatchNorm2d, Flatten, BatchNorm1d
from torch.nn.functional import hardsigmoid


class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, filter_size, num_filters, stride, padding, num_groups=1):
        super(ConvBNLayer, self).__init__()
        self._conv = Conv2d(in_channels=num_channels,
                            out_channels=num_filters,
                            kernel_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=num_groups)

        self._batch_norm = BatchNorm2d(num_filters)

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class DepthwiseSeparable(nn.Module):
    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3, padding=1):
        super(DepthwiseSeparable, self).__init__()
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels,
                                           num_filters=int(num_filters1 * scale),
                                           filter_size=dw_size,
                                           stride=stride,
                                           padding=padding,
                                           num_groups=int(num_groups * scale))
        self._pointwise_conv = ConvBNLayer(num_channels=int(num_filters1 * scale),
                                           filter_size=1,
                                           num_filters=int(num_filters2 * scale),
                                           stride=1,
                                           padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._pointwise_conv(y)
        return y


class DepthwiseSeparableSE(nn.Module):
    def __init__(self, num_channels, num_filters1, num_filters2, num_groups, stride, scale, dw_size=3, padding=1):
        super(DepthwiseSeparableSE, self).__init__()
        self._depthwise_conv = ConvBNLayer(num_channels=num_channels,
                                           num_filters=int(num_filters1 * scale),
                                           filter_size=dw_size,
                                           stride=stride,
                                           padding=padding,
                                           num_groups=int(num_groups * scale))
        self._se = SEModule(int(num_filters1 * scale))
        self._pointwise_conv = ConvBNLayer(num_channels=int(num_filters1 * scale),
                                           filter_size=1,
                                           num_filters=int(num_filters2 * scale),
                                           stride=1,
                                           padding=0)

    def forward(self, inputs):
        y = self._depthwise_conv(inputs)
        y = self._se(y)
        y = self._pointwise_conv(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(in_channels=channel,
                            out_channels=channel // reduction,
                            kernel_size=(1, 1))
        self.conv2 = Conv2d(in_channels=channel // reduction,
                            out_channels=channel,
                            kernel_size=(1, 1))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = hardsigmoid(outputs)
        return torch.multiply(inputs, outputs)


class MobileFaceNet(nn.Module):
    def __init__(self, in_channels=3, scale=1.0):
        super().__init__()
        self.scale = scale
        self.block_list = []

        self.conv1 = ConvBNLayer(num_channels=in_channels,
                                 filter_size=3,
                                 num_filters=int(32 * scale),
                                 stride=2,
                                 padding=1)

        conv2_1 = DepthwiseSeparable(num_channels=int(32 * scale),
                                     num_filters1=32,
                                     num_filters2=64,
                                     num_groups=32,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_1)

        conv2_2 = DepthwiseSeparable(num_channels=int(64 * scale),
                                     num_filters1=64,
                                     num_filters2=128,
                                     num_groups=64,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv2_2)

        conv3_1 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=128,
                                     num_groups=128,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv3_1)

        conv3_2 = DepthwiseSeparable(num_channels=int(128 * scale),
                                     num_filters1=128,
                                     num_filters2=256,
                                     num_groups=128,
                                     stride=2,
                                     scale=scale)
        self.block_list.append(conv3_2)

        conv4_1 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=256,
                                     num_groups=256,
                                     stride=1,
                                     scale=scale)
        self.block_list.append(conv4_1)

        conv4_2 = DepthwiseSeparable(num_channels=int(256 * scale),
                                     num_filters1=256,
                                     num_filters2=512,
                                     num_groups=256,
                                     stride=2,
                                     scale=scale)
        self.block_list.append(conv4_2)

        for _ in range(5):
            conv5 = DepthwiseSeparableSE(num_channels=int(512 * scale),
                                         num_filters1=512,
                                         num_filters2=512,
                                         num_groups=512,
                                         stride=1,
                                         dw_size=5,
                                         padding=2,
                                         scale=scale)
            self.block_list.append(conv5)

        conv5_6 = DepthwiseSeparableSE(num_channels=int(512 * scale),
                                       num_filters1=512,
                                       num_filters2=1024,
                                       num_groups=512,
                                       stride=2,
                                       dw_size=5,
                                       scale=scale)
        self.block_list.append(conv5_6)

        conv6 = DepthwiseSeparableSE(num_channels=int(1024 * scale),
                                     num_filters1=1024,
                                     num_filters2=1024,
                                     num_groups=1024,
                                     stride=2,
                                     dw_size=5,
                                     scale=scale)
        self.block_list.append(conv6)

        self.block_list = nn.Sequential(*self.block_list)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = Flatten()
        self.linear = Linear(in_features=int(1024 * scale), out_features=int(1024 * scale))
        self.bn = BatchNorm1d(num_features=int(1024 * scale))

    def forward(self, inputs):
        y = self.conv1(inputs)
        y = self.block_list(y)
        y = self.pool(y)
        y = self.flatten(y)
        y = self.linear(y)
        y = self.bn(y)
        return y

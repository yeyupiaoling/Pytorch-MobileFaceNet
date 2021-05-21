import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class ArcMarginNet(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            num_classes: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, feature_dim, class_dim, s=64.0, m=0.50):
        super(ArcMarginNet, self).__init__()
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(class_dim, feature_dim))
        nn.init.xavier_uniform_(self.weight)
        self.class_dim = class_dim
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, feature, label):
        cos_theta = F.linear(F.normalize(feature), F.normalize(self.weight))
        sin_theta = torch.sqrt(torch.clip(1.0 - torch.pow(cos_theta, 2), min=0, max=1))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)
        one_hot = torch.nn.functional.one_hot(label, self.class_dim)
        output = (one_hot * cos_theta_m) + ((1.0 - one_hot) * cos_theta)
        output *= self.s
        return output

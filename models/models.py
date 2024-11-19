# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM
from .cnn_mamba_parts import vss_small
from .convunext import convnext_tiny
from .Resnet import resnet50
# from .Res2Net import res2net50_v1b_26w_4s
# from .Res2Net import res2net50_v1b
from .unet_parts import *
from .mokuai import *
import matplotlib.pyplot as plt
import cv2
logger = logging.getLogger(__name__)

class MBUnet(nn.Module):
    def __init__(self,  img_size=256, num_classes=1, zero_head=False, vis=False):
        super(MBUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.backbone1 = vss_small(pretrained=True)
        self.backbone2 = convnext_tiny(pretrained=True)
        # self.resnet = res2net50_v1b_26w_4s(pretrained=True)
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

#============================================================================
        XX2 = self.backbone2(x)
        cnn_stage1, cnn_stage2, cnn_stage3, cnn_stage4 = XX2 # 局部信息
        feature = []
        feature.append(cnn_stage1)
        feature.append(cnn_stage2)
        feature.append(cnn_stage3)
        feature.append(cnn_stage4)
        # # print(len(feature))
#============================================================================
        logits = self.backbone1(x,feature) # 长程信息
#============================================================================
        if self.num_classes == 1: return torch.sigmoid(logits)
        return logits
    

if __name__ == "__main__":
    model = MBUnet().to('cuda')
    int = torch.randn(16,3,256,256).cuda()
    out = model(int)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)
    # model = DuAT().to('cuda')
    # from torchinfo import summary
#    summary(model, (1, 3, 352, 352))
    from thop import profile
    import torch
    input = torch.randn(1, 3, 352, 352).to('cuda')
    macs, params = profile(model, inputs=(input,))
    print('macs:', macs / 1000000000)
    print('params:', params / 1000000)
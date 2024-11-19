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
# from .Resnet import resnet50
# from .Res2Net import res2net50_v1b_26w_4s
# from .Res2Net import res2net50_v1b
from .unet_parts import *
from .mokuai import *
import matplotlib.pyplot as plt
import cv2
logger = logging.getLogger(__name__)

class Two_encode(nn.Module):
    def __init__(self,  img_size=256, num_classes=1, zero_head=False, vis=False):
        super(Two_encode, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.backbone1 = vss_small(pretrained=True)
        self.backbone2 = convnext_tiny(pretrained=True)
        self.SBA1 = SBA1(96)
        self.SBA2 = SBA1(192)
        self.SBA3 = SBA1(384)
        self.SBA4 = SBA1(768)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.GELU()
        )
        # self.up1_ = nn.Conv2d(96, 1, 1)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(96),
            nn.GELU()
        )
        self.up2_ = nn.Conv2d(192, 96, 1)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(192),
            nn.GELU()
        )
        self.up3_ = nn.Conv2d(384, 192, 1)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.GELU()
        )
        self.up4_ = nn.Conv2d(768, 384, 1)
    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)


        XX2 = self.backbone2(x)
        cnn_stage1, cnn_stage2, cnn_stage3, cnn_stage4 = XX2 # 局部信息

#============================================================================
        x, x_downsample  = self.backbone1(x) # 长程信息
        # print(logits.shape)
#============================================================================
        x_downsample[0] = self.SBA1(cnn_stage1,x_downsample[0].permute(0, 3, 1, 2))
        x_downsample[1] = self.SBA2(cnn_stage2,x_downsample[1].permute(0, 3, 1, 2))
        x_downsample[2] = self.SBA3(cnn_stage3,x_downsample[2].permute(0, 3, 1, 2))
        x_downsample[3] = self.SBA4(cnn_stage4,x_downsample[3].permute(0, 3, 1, 2))
#============================================================================================
        
        x = x.permute(0, 3, 1, 2)
        # print(x.shape)
        # print(x_downsample[3].shape)
        up4 = self.up4(x)
        up4 = torch.cat([up4, x_downsample[2]], dim=1)
        up4 = self.up4_(up4)
        # print(up4.shape)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, x_downsample[1]], dim=1)
        up3 = self.up3_(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, x_downsample[0]], dim=1)
        up2 = self.up2_(up2)

        logits = self.up1(up2)
        # print(out.shape)
        # logits = self.out(out)
#============================================================================================
        # print(x1_5.shape) 
        # print(mamba_stage1.shape)
        # print(cnn_stage1.shape)

        # logits = self.SBA1(edge,logits)
        if self.num_classes == 1: return torch.sigmoid(logits)
        return logits
    

if __name__ == "__main__":
    model = Two_encode().to('cuda')
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
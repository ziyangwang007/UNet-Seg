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

logger = logging.getLogger(__name__)

def vss_small(pretrained=True,**kwargs):
    model = VSSM(
                    patch_size=4,
                    in_chans=3,
                    num_classes=1,
                    embed_dim=96,
                    depths=[2,2,2,2],
                    mlp_ratio=4.,
                    drop_rate=0.0,
                    drop_path_rate=0.2,
                    patch_norm=True,
                    use_checkpoint=False)
    if pretrained:
        pretrained_path = 'pretrained_ckpt/vmamba_small_e238_ema.pth'
        print("pretrained_path:{}".format(pretrained_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        if "model"  not in pretrained_dict:
            print("---start load pretrained modle by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = model.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return
        pretrained_dict = pretrained_dict['model']
        print("---start load pretrained modle of vmamba encoder---")

        model_dict = model.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]
        msg = model.load_state_dict(full_dict, strict=False)
        # print(msg)
    else:
        print("none pretrain")
    return model

if __name__ == "__main__":
    model = vss_small(pretrained=True).to('cuda')
    int1 = torch.randn(16,3,256,256).cuda()
    int2 = torch.randn(16,3,256,256).cuda()
    X = model(int1,int2)
    # print(out[0].shape)
    # print(out[1].shape)
    # print(out[2].shape)
    # print(out[3].shape)
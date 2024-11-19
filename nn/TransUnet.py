# import torch
# from self_attention_cv.transunet import TransUnet
# a = torch.rand(2, 3, 128, 128)
# model = TransUnet(in_channels=3, img_dim=128, vit_blocks=8,
# vit_dim_linear_mhsa_block=512, classes=2)
# y = model(a) # [2, 5, 128, 128]
# print(y.shape)
import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchsummary import summary
def get_transNet(n_classes):
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes)
    return net


if __name__ == '__main__':
    net = get_transNet(2)
    img = torch.randn((2, 3, 480, 480))
    segments = net(img)
    # print('111',segments['out1'].size())
    summary(net, input_size=(3, 256, 256))
    # for edge in edges:
    #     print(edge.size())
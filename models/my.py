import torch
import torch.nn as nn
import torch.nn.functional as F

# from core.convnext import convnext_base
from models.vss import vss_base
from models.unet_parts import *


class My_Net(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(BCU_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone = vss_base(pretrained=True)
        self.bilinear = bilinear
        self.pool = nn.MaxPool2d((2, 2))
        self.conv = DoubleConv(n_channels, 32)
        self.down1_1 = Down(32, 64)
        self.down1_2 = Down(64, 128)
        self.down1_3 = Down(128, 256)
        self.down1_4 = Down(256, 512)

        self.up1_1 = Up(512, 256, bilinear)
        self.up1_2 = Up(256, 128, bilinear)
        self.up1_3 = Up(128, 64, bilinear)
        self.up1_4 = Up(64, 32, bilinear)


        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # self.up1_ = nn.Conv2d(128, 64, 1)
        self.up1_ = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=True),
            nn.BatchNorm2d(64),
            nn.GELU())

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.up2_ = nn.Conv2d(256, 128, 1)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.up3_ = nn.Conv2d(512, 256, 1)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        self.up4_ = nn.Conv2d(1024, 512, 1)

        self.cls_seg = nn.Conv2d(64, 1, 1)

        self.out = OutConv(32, 1)

    def forward(self, x):
        xx = self.backbone(x)
        stage1, stage2, stage3, stage4 = xx
        up4 = self.up4(stage4)
        up4 = torch.cat([up4, stage3], dim=1)
        up4 = self.up4_(up4)

        up3 = self.up3(up4)
        up3 = torch.cat([up3, stage2], dim=1)
        up3 = self.up3_(up3)

        up2 = self.up2(up3)
        up2 = torch.cat([up2, stage1], dim=1)
        up2 = self.up2_(up2)

        out = self.up1(up2)
        out = self.up1_(out)
        out2 = self.cls_seg(out)



        x1_1 = self.conv(x)
        x1_2 = self.down1_1(x1_1)
        x1_3 = self.down1_2(x1_2)
        x1_4 = self.down1_3(x1_3)
        x1_5 = self.down1_4(x1_4)
        x1_6 = self.up1_1(x1_5, x1_4)
        x1_7 = self.up1_2(x1_6, x1_3)
        x1_8 = self.up1_3(x1_7, x1_2)
        x1_9 = self.up1_4(x1_8, x1_1)
        out1 = self.out(x1_9)

        # print(out1.shape)
        out = out1 + out2
        return torch.sigmoid(out)
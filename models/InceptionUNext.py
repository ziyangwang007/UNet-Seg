from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from einops import rearrange 
import math
import torchvision
def channel_shuffle(x, groups: int):
 
    # 在pytorch中所得到的tensor通道排列顺序
    batch_size, num_channels, height, width = x.size()
    # 将channels划分为groups组
    channels_per_group = num_channels // groups
 
    # reshape
    # [batch_size, num_channels, height, width] → [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)
 
    # 将维度1（groups）和维度2（channels_per_group）的信息进行交换
    # transpose后，tensor在内存中的存储顺序不是连续的，contiguous可以将数据转化为连续的数据
    x = torch.transpose(x, 1, 2).contiguous()
 
    # flatten
    x = x.view(batch_size, -1, height, width)
 
    return x

class DeformConv(nn.Module):

    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
class deformable_LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        gc1 = int(dim * 0.25)
        gc2 = int(dim * 0.5)
        gc3 = int(dim * 0.25)
        self.conv0 = DeformConv(gc1, groups=gc1, kernel_size=(3,3), padding=1)
        self.dw2 = nn.Conv2d(gc3, gc3, kernel_size=3, padding='same',dilation = 2)
        self.dw = nn.Conv2d(gc2, gc2, kernel_size = 7, padding = 3, stride=1, groups=gc2)
        self.split_indexes = (gc1, gc2, gc3)
        # self.split_indexes = (gc3, gc1)

    def forward(self, x):
        x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)  
        # x_w, x_h = torch.split(x, self.split_indexes, dim=1)

        return torch.cat((self.conv0(x_hw), self.dw(x_w), self.dw2(x_h)), dim=1,)
        # return torch.cat((self.dw(x_w), self.dw2(x_h)), dim=1,)
class Conv(nn.Module):
    def __init__(self, dim):
        super(Conv, self).__init__()
        # self.dwconv = DeformConv(dim, groups=dim, kernel_size=(3,3), padding=1)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding='same',dilation = 2)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.dwconv = deformable_LKA(dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.act2 = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = channel_shuffle(x,2)
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.norm2(x)
        x = self.act2(residual + x)

        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, layer_num=1):
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        super(Down, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            *layers
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, layer_num=1):
        super(Up, self).__init__()
        C = in_channels // 2
        self.norm = nn.BatchNorm2d(C)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        layers = nn.ModuleList()
        for i in range(layer_num):
            layers.append(Conv(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.norm(x1)
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        #attention
        # B, C, H, W = x1.shape
        # x1 = x1.permute(0, 2, 3, 1)
        # x2 = x2.permute(0, 2, 3, 1)
        # gate = self.gate(x1).reshape(B, H, W, 3, C).permute(3, 0, 1, 2, 4)
        # g1, g2, g3 = gate[0], gate[1], gate[2]
        # x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        # x2 = self.linear2(x2)
        # x1 = x1.permute(0, 3, 1, 2)
        # x2 = x2.permute(0, 3, 1, 2)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class mynet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(mynet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_c, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.BatchNorm2d(base_c),
            nn.GELU(),
            Conv(base_c)
        )
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8, layer_num=3)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return logits

if __name__ == '__main__':
    model = mynet(in_channels=3, num_classes=1, base_c=32).to('cuda')
    summary(model, input_size=(3, 256, 256))

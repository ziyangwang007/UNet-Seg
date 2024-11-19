import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
############交叉融合模块#########################################

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)
class SBA(nn.Module):

    def __init__(self,input_dim = 64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim//2, input_dim//2, 1)
        self.d_in2 = BasicConv2d(input_dim//2, input_dim//2, 1)       
                

        self.conv = nn.Sequential(BasicConv2d(input_dim, input_dim, 3,1,1))
        self.fc1 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, H_feature, L_feature):
        
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)


        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
     
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature,L_feature], dim=1))
        return out

#=======================================================================
class SBA1(nn.Module):

    def __init__(self,input_dim = 64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim//2, 16, 1)
        self.d_in2 = BasicConv2d(input_dim//2, 16, 1)       
                

        self.conv = nn.Sequential(BasicConv2d(2*input_dim, input_dim, 3,1,1))
        self.fc1 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
        self.weight = nn.Conv2d(16*2, 2, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(4, 4))

    def forward(self, H_feature, L_feature):

        # H_feature = F.interpolate(H_feature, size=L_feature.size()[2:], mode='bilinear', align_corners=False)
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        L_feature1 = self.fc1(L_feature)
        H_feature1 = self.fc2(H_feature)
        
 
        L_feature1 = self.d_in1(L_feature1)
        H_feature1 = self.d_in2(H_feature1)
  

        feature = torch.cat((L_feature1,H_feature1),dim=1)
        #======================GCN=================================
        # print(feature.shape)
        N,C,H,W = feature.size()
        feature1 = self.priors(feature)
        # print(feature.shape)
        feature1 = feature1.view(N, 32, -1)
        # print(x.shape)
        # print(feature1.permute(0, 2, 1).shape)
        h = self.conv1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - feature1
        h = self.relu(self.conv2(h))
        # print(h.shape)
        h = h.view(N, C, 4, 4)
        # print(h.shape)
        h= Upsample(h, size = L_feature.size()[2:])
        # h = h.view(N, C, H, W)
        #======================GCN=================================
        weight = self.weight(h)
        # weight = self.gcn(weight)
        weighted = self.Sigmoid(weight)
        # L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        # H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        # H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out1 = L_feature + L_feature * weighted[:,0:1,:,:] + (1 - weighted[:,0:1,:,:]) * H_feature*weighted[:,1:,:,:]
        out2 = H_feature + H_feature * weighted[:,1:,:,:] + (1 - weighted[:,1:,:,:]) * L_feature*weighted[:,0:1,:,:]
        # H_feature = Upsample(H_feature, size = L_feature.size()[2:])

        out = self.conv(torch.cat([out1,out2], dim=1))
        # print(out.shape)
        return out    
    
#=======================================================================
class SBA2(nn.Module):

    def __init__(self,input_dim = 64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = BasicConv2d(input_dim//2, 16, 1)
        self.d_in2 = BasicConv2d(input_dim//2, 16, 1)       
                

        self.conv = nn.Sequential(BasicConv2d(2*input_dim, input_dim, 3,1,1))
        self.fc1 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(input_dim, input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
        self.weight = nn.Conv2d(16*2, 2, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv1d(16, 16, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=1, bias=False)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(4, 4))

    def forward(self, H_feature, L_feature):

        # H_feature = F.interpolate(H_feature, size=L_feature.size()[2:], mode='bilinear', align_corners=False)
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        
        H_feature = torch.fft.fft2(H_feature, dim=(-2,-1)).real
        L_feature = torch.fft.fft2(L_feature, dim=(-2,-1)).real

        L_feature1 = self.fc1(L_feature)
        H_feature1 = self.fc2(H_feature)
        
        L_feature1 = self.d_in1(L_feature1)
        H_feature1 = self.d_in2(H_feature1)
  
        feature = torch.cat((L_feature1,H_feature1),dim=1)
        #======================GCN=================================
        # # print(feature.shape)
        # N,C,H,W = feature.size()
        # feature1 = self.priors(feature)
        # # print(feature.shape)
        # feature1 = feature1.view(N, 32, -1)
        # # print(x.shape)
        # # print(feature1.permute(0, 2, 1).shape)
        # h = self.conv1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        # h = h - feature1
        # h = self.relu(self.conv2(h))
        # # print(h.shape)
        # h = h.view(N, C, 4, 4)
        # # print(h.shape)
        # h= Upsample(h, size = L_feature.size()[2:])
        # # h = h.view(N, C, H, W)
        #======================GCN=================================
        #======================frequency=================================

        # H_feature = torch.fft.fft2(H_feature, dim=(-2,-1))
        # L_feature = torch.fft.fft2(L_feature, dim=(-2,-1))

        #======================frequency=================================
        weight = self.weight(feature)
        # weight = self.gcn(weight)
        weighted = self.Sigmoid(weight)
        # L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        # H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        # H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out1 = L_feature + L_feature * weighted[:,0:1,:,:] + (1 - weighted[:,0:1,:,:]) * H_feature*weighted[:,1:,:,:]
        out2 = H_feature + H_feature * weighted[:,1:,:,:] + (1 - weighted[:,1:,:,:]) * L_feature*weighted[:,0:1,:,:]
        # H_feature = Upsample(H_feature, size = L_feature.size()[2:])

        out1 = torch.fft.ifft2(out1, dim=(-2, -1))
        out2 = torch.fft.ifft2(out2, dim=(-2, -1))
        out = self.conv(torch.cat([out1.real,out2.real], dim=1))
        # print(out.shape)
        return out  
#=======================================================================
class GCN_channel(nn.Module):
    def __init__(self, channel):
        super(GCN_channel, self).__init__()  
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.channel = channel
        # self.para = torch.nn.Parameter(torch.ones((1,channel,h,w), dtype = torch.float32))
        # self.adj = torch.nn.Parameter(torch.ones((channel,channel), dtype = torch.float32))
             
    def forward(self, x):
        device = 'cuda:0'
        b, c, H, W = x.size()
        fea_matrix = x.view(b,c,H*W)
        c_adj = self.avg_pool(x).view(b,c)
        para = torch.nn.Parameter(torch.ones((1,self.channel,H,W), dtype = torch.float32)).to(device)
        adj = torch.nn.Parameter(torch.ones((self.channel,self.channel), dtype = torch.float32)).to(device)
        m = torch.ones((b,c,H,W), dtype = torch.float32)

        for i in range(0,b):

            t1 = c_adj[i].unsqueeze(0)
            t2 = t1.t()
            c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1-t2)-0.5)-0.5)*2
            c_adj_s = (c_adj_s.t() + c_adj_s)/2

            output0 = torch.mul(torch.mm(adj*c_adj_s,fea_matrix[i]).view(1,c,H,W),para)

            m[i] = output0

        output = torch.nn.functional.relu(m.cuda())

        return output
    # def to_matrix(self, x):
    #     n, c, h, w = x.size()
    #     x = x.view(n, c, -1)
    #     return x 

class GCN1(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN1, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h
#######################边界损失模块######################################
def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss


# class PagFM(nn.Module):
#     def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False, BatchNorm=nn.BatchNorm2d):
#         super(PagFM, self).__init__()
#         self.with_channel = with_channel
#         self.after_relu = after_relu
#         self.f_x = nn.Sequential(
#                                 nn.Conv2d(in_channels, mid_channels, 
#                                           kernel_size=1, bias=False),
#                                 BatchNorm(mid_channels)
#                                 )
#         self.f_y = nn.Sequential(
#                                 nn.Conv2d(in_channels, mid_channels, 
#                                           kernel_size=1, bias=False),
#                                 BatchNorm(mid_channels)
#                                 )
#         if with_channel:
#             self.up = nn.Sequential(
#                                     nn.Conv2d(mid_channels, in_channels, 
#                                               kernel_size=1, bias=False),
#                                     BatchNorm(in_channels)
#                                    )
#         if after_relu:
#             self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x, y):
#         input_size = x.size()
#         if self.after_relu:
#             y = self.relu(y)
#             x = self.relu(x)
        
#         y_q = self.f_y(y)
#         y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
#                             mode='bilinear', align_corners=False)
#         x_k = self.f_x(x)
        
#         if self.with_channel:
#             sim_map = torch.sigmoid(self.up(x_k * y_q))
#         else:
#             sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        
#         y = F.interpolate(y, size=[input_size[2], input_size[3]],
#                             mode='bilinear', align_corners=False)
#         x = (1-sim_map)*x + sim_map*y
        
#         return x

class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = nn.Conv2d(96, 64, 1)
        self.reduce4 = nn.Conv2d(768, 256,1)
        self.block = nn.Sequential(
            BasicConv2d(256 + 64, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

class ASFF(nn.Module): 
    def __init__(self, level, rfb=False, vis=False): 
        super(ASFF, self).__init__() 
        self.level = level 
        self.dim = [512, 256, 256] 
        self.inter_dim = self.dim[self.level] 
        # 每个level融合前，需要先调整到一样的尺度
        if level==0: 
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2) 
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2) 
            self.expand = add_conv(self.inter_dim, 1024, 3, 1) 
        elif level==1: 
            self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1) 
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2) 
            self.expand = add_conv(self.inter_dim, 512, 3, 1) 
        elif level==2: 
           self.compress_level_0 = add_conv(512, self.inter_dim, 1, 1) 
           self.expand = add_conv(self.inter_dim, 256, 3, 1) 
        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory 

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1) 
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1) 
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1) 

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0) 
        self.vis= vis 
       
    def forward(self, x_level_0, x_level_1, x_level_2): 
        if self.level==0: 
           level_0_resized = x_level_0 
           level_1_resized = self.stride_level_1(x_level_1) 
 
           level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1) 
           level_2_resized = self.stride_level_2(level_2_downsampled_inter) 
 
        elif self.level==1: 
           level_0_compressed = self.compress_level_0(x_level_0) 
           level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest') 
           level_1_resized =x_level_1 
           level_2_resized =self.stride_level_2(x_level_2) 
        elif self.level==2: 
           level_0_compressed = self.compress_level_0(x_level_0) 
           level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest') 
           level_1_resized =F.interpolate(x_level_1, scale_factor=2, mode='nearest') 
           level_2_resized =x_level_2 
 
        level_0_weight_v = self.weight_level_0(level_0_resized) 
        level_1_weight_v = self.weight_level_1(level_1_resized) 
        level_2_weight_v = self.weight_level_2(level_2_resized) 
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1) 
        # 学习的3个尺度权重
        levels_weight = self.weight_levels(levels_weight_v) 
        levels_weight = F.softmax(levels_weight, dim=1) 
        # 自适应权重融合
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+level_1_resized * levels_weight[:,1:2,:,:]+level_2_resized * levels_weight[:,2:,:,:] 
        out = self.expand(fused_out_reduced) 
  
        if self.vis: 
           return out, levels_weight, fused_out_reduced.sum(dim=1) 
        else: 
          return out 

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        n, c, H, W = x.size()
        x = x.view(n, c, -1)
        print(x.shape)
        print(x.permute(0, 2, 1).shape)
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        h = h.view(n, c, H, W)
        # print(h.shape)
        return h


class CGRModule(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):
        super(CGRModule, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = (mids) * (mids)
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = nn.BatchNorm2d(num_in)


    def forward(self, x, edge):
        edge = F.upsample(edge, (x.size()[-2], x.size()[-1]))

        n, c, h, w = x.size()
        edge = torch.nn.functional.softmax(edge, dim=1)


        # Construct projection matrix
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge # torch.Size([3, 96, 64, 64])
        print(x_mask.shape)
        a = self.priors(x_mask)[:, :, 1:-1, 1:-1]
        print(a.shape) # torch.Size([3, 96, 4, 4])
        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        # print(x_proj.reshape(n, self.num_s, -1).shape)
        print(x_anchor.shape) # torch.Size([3, 96, 16])
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        print(x_proj_reshaped.shape) # torch.Size([3, 16, 4096])
        x_proj_reshaped = torch.nn.functional.softmax(x_proj_reshaped, dim=1)
        x_rproj_reshaped = x_proj_reshaped

        # print(x_rproj_reshaped.shape)
      

        # Project and graph reason
        print(x_state_reshaped.shape)
        print(x_proj_reshaped.permute(0, 2, 1).shape)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        print(x_n_state.shape)
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        print(x_n_rel.shape)

        # Reproject
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)       #x_n_rel   ###没有gcn
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        out = x + self.blocker(self.conv_extend(x_state))
        # print(out.shape)
        return out
def get_seqlen_and_mask(input_resolution, window_size):
    attn_map = F.unfold(torch.ones([1, 1, input_resolution[0], input_resolution[1]]), window_size,
                        dilation=1, padding=(window_size // 2, window_size // 2), stride=1)
    attn_local_length = attn_map.sum(-2).squeeze().unsqueeze(-1)
    attn_mask = (attn_map.squeeze(0).permute(1, 0)) == 0
    return attn_local_length, attn_mask
class AggregatedAttention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.sr_ratio = sr_ratio

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.pool_H, self.pool_W = input_resolution[0] // self.sr_ratio, input_resolution[1] // self.sr_ratio
        self.pool_len = self.pool_H * self.pool_W

        self.unfold = nn.Unfold(kernel_size=window_size, padding=window_size // 2, stride=1)
        self.temperature = nn.Parameter(torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)) #Initialize softplus(temperature) to 1/0.24.

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #Components to generate pooled features.
        self.pool = nn.AdaptiveAvgPool2d((self.pool_H, self.pool_W))
        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        # relative bias for local features
        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.local_len), mean=0,
                                  std=0.0004))

        # Generate padding_mask && sequnce length scale
        local_seq_length, padding_mask = get_seqlen_and_mask(input_resolution, window_size)
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(local_seq_length.numpy() + self.pool_len)),
                             persistent=False)
        self.register_buffer("padding_mask", padding_mask, persistent=False)

        # dynamic_local_bias:
        self.learnable_tokens = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02))
        self.learnable_bias = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape

        #Generate queries, normalize them with L2, add query embedding, and then magnify with sequence length scale and temperature.
        #Use softplus function ensuring that the temperature is not lower than 0.
        q_norm=F.normalize(self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3),dim=-1)
        q_norm_scaled = (q_norm + self.query_embedding) * F.softplus(self.temperature) * self.seq_length_scale

        # Generate unfolded keys and values and l2-normalize them
        k_local, v_local = self.kv(x).chunk(2, dim=-1)
        k_local = F.normalize(k_local.reshape(B, N, self.num_heads, self.head_dim), dim=-1).reshape(B, N, -1)
        kv_local = torch.cat([k_local, v_local], dim=-1).permute(0, 2, 1).reshape(B, -1, H, W)
        k_local, v_local = self.unfold(kv_local).reshape(
            B, 2 * self.num_heads, self.head_dim, self.local_len, N).permute(0, 1, 4, 2, 3).chunk(2, dim=1)

        # Compute local similarity
        attn_local = ((q_norm_scaled.unsqueeze(-2) @ k_local).squeeze(-2) \
                      + self.relative_pos_bias_local.unsqueeze(1)).masked_fill(self.padding_mask, float('-inf'))

        # Generate pooled features
        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = self.pool(self.act(self.sr(x_))).reshape(B, -1, self.pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        # Generate pooled keys and values
        kv_pool = self.kv(x_).reshape(B, self.pool_len, 2 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        #Use MLP to generate continuous relative positional bias for pooled features.
        pool_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                    relative_pos_index.view(-1)].view(-1, N, self.pool_len)
        # Compute pooled similarity
        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        # Concatenate local & pooled similarity matrices and calculate attention weights through the same Softmax
        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)

        #Split the attention weights and separately aggregate the values of local & pooled features
        attn_local, attn_pool = torch.split(attn, [self.local_len, self.pool_len], dim=-1)
        x_local = (((q_norm @ self.learnable_tokens) + self.learnable_bias + attn_local).unsqueeze(-2) @ v_local.transpose(-2, -1)).squeeze(-2)
        x_pool = attn_pool @ v_pool
        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)

        #Linear projection and output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W
if __name__ == '__main__':
    # a = torch.zeros(2,64,64)
    # # a[:,5,:] = 1
    # pre = torch.randn(2,1,16,16)
    
    # Loss_fc = SurfaceLoss(idc=[0,1])
    # loss = Loss_fc(pre, a.to(torch.uint8))
    x = torch.randn(3,96,48,48).cuda()
    y = torch.randn(3,96,128,128)
    # relative_pos_index, relative_coords_table = get_relative_position_cpb(query_size=to_2tuple(48),
    #                                                                             key_size=to_2tuple(48),
    #                                                                             pretrain_size=to_2tuple(48))
    # patch_embed = OverlapPatchEmbed(patch_size=3,
    #                             stride=2,
    #                             in_chans=96,
    #                             embed_dim=96).cuda()  
    # x= patch_embed(x)    
    # print(relative_pos_index.shape)   
    # print(relative_coords_table.shape)                                                                      
    # print(x[2])
    # model = AggregatedAttention(96,(48,48)).cuda()
    # out = model(x[0],24,24,relative_pos_index,relative_coords_table)
    # print(out.shape)
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import numpy as np
# 这里以上述创建的单数据为例子
# data = np.array([
#                 [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]],
#                 [[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
#                 [[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
#                 [[4,4,4],[4,4,4],[4,4,4],[4,4,4],[4,4,4]],
#                 [[5,5,5],[5,5,5],[5,5,5],[5,5,5],[5,5,5]]
#         ],dtype='uint8')

# 将数据转为C,W,H，并归一化到[0，1]
data = Image.open('data/isic2017/train/images/ISIC_0000000.jpg')
data =  np.array(data)
data = transforms.ToTensor()(data)
# 需要对数据进行扩维，增加batch维度
data = torch.unsqueeze(data,0)

nb_samples = 0.
#创建3维的空列表
channel_mean = torch.zeros(3)
channel_std = torch.zeros(3)
print(data.shape)
N, C, H, W = data.shape[:4]
data = data.view(N, C, -1)     #将w,h维度的数据展平，为batch，channel,data,然后对三个维度上的数分别求和和标准差
print(data.shape)
#展平后，w,h属于第二维度，对他们求平均，sum(0)为将同一纬度的数据累加
channel_mean += data.mean(2).sum(0)  
#展平后，w,h属于第二维度，对他们求标准差，sum(0)为将同一纬度的数据累加
channel_std += data.std(2).sum(0)
#获取所有batch的数据，这里为1
nb_samples += N
#获取同一batch的均值和标准差
channel_mean /= nb_samples
channel_std /= nb_samples
print(channel_mean, channel_std)
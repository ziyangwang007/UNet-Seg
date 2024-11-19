# import pandas as pd
import csv
import xlwt 
# import openpyxl
import numpy as np
# import pandas as pd
import math
import numpy as np
import torch
def entropy_weight(x):
    # 计算每个指标的熵值
    b, c, m, n = x.shape
    print(n)
    e = np.zeros((1, n))
    for j in range(n):
        p = x[:,:,:, j] / x[:,:,:, j].sum()
        e[0][j] = - (p * np.log(p)).sum()

    # 计算每个指标的权重
    w = np.zeros((1, n))
    for j in range(n):
        w[0][j] = (1 - e[0][j]) / ((1 - e).sum())

    return w

def topsis(x, w):
    # 将x归一化处理
    b, c, m, n = x.shape
    x_norm = np.zeros((m, n))
    for j in range(n):
        x_norm[:, j] = x[:, j] / np.sqrt((x[:, j]**2).sum())

    # 计算加权后的矩阵
    x_weighted = np.zeros((m, n))
    for j in range(n):
        x_weighted[:, j] = w[0][j] * x_norm[:, j]

    # 计算最优解和最劣解
    max_vec = x_weighted.max(axis=0)
    min_vec = x_weighted.min(axis=0)

    # 计算每个评价对象与最优解和最劣解的距离
    d_plus = np.sqrt(((x_weighted - max_vec)**2).sum(axis=1))
    d_minus = np.sqrt(((x_weighted - min_vec)**2).sum(axis=1))

    # 计算得分
    score = d_minus / (d_minus + d_plus)

    return score

# 示例数据
# x = np.array([0.4,0.8,0.3])
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(x.shape)
int1 = torch.randn(16,2,256,256)
# int2 = torch.randn(16,16,256,256)
# int = torch.cat((int1,int2),dim=1)
# print(int.shape)
# 计算熵权法得到的权重
w = entropy_weight(int1)
print(w)

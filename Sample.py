# pip install opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple/
# pip install numpy

#=======================class1=====================================
# 相对路径：
# import cv2
# # img = cv2.imread('./b.jpg')
# # print(np.dtype(img))
# # print(img.shape)
# # print(img.size)
# # print(img.size())
# import numpy as np
# #绝对路径
# img = cv2.imread('/mnt/VM-UNet/b.jpg')
# print(type(img))
# print(img.shape)
# print(img.size)
#=======================class2=====================================
import numpy as np 
import cv2 
# img1 = np.random.randint(0,256,size=[3,4], dtype=np.uint8)
# print(img1) #[3,4]
# img2 = np.random.randint(0,256,size=[4,3],dtype=np.uint8)
# print(img2) #[4,3]
# result1 = np.dot(img1,img2)
# print(result1)
# [[175 212  13 115] img1     
#  [ 35 193  25 154]
#  [ 77 249 212 215]]

# [[  1 173 245]     img2  175 33072 3237 54769
#  [156 164  69]
#  [249  12 220]
#  [159 126 199]]

# [[241  73  48]      result1
#  [182  67 182]
#  [198  79  31]]


# 1*3 + 2*1 = 5,10
            #   13,22

# img3 = np.random.randint(0, 256, size=[4,4], dtype=np.uint8)
# img4 = np.random.randint(0, 256, size=[4,4], dtype=np.uint8)
# print(img3)
# print(img4)
# result2 = cv2.divide(img3,img4)
# print(result2)


# img3 = np.random.randint(0, 256, size=[4,4], dtype=np.uint8)
# img4 = np.random.randint(0, 256, size=[4,4], dtype=np.uint8)
# print(img3)
# print(img4)
# img3 =[51]
# img3 = np.array(img3)
# img4 = [144]
# img4 = np.array(img4)
# result2 = cv2.bitwise_xor(img3,img4)
# print(result2)
# [[245  51  63 187] img3           128  64 32 16 8 4 2 1
#  [ 50  73  15 165]                0     0  1  1 0 0 1 1            
#  [ 56 129 103  97]
#  [123 253  18 117]]

# [[180 144 221 151] img4           1     0  0  1 0 0 0 0
#  [219 183 248 174]
#  [209  84  28 174]
#  [ 56  52 188 194]]

# [[180  16  29 147] result2        1     0  1  0 0 0 1 1 = 163
#  [ 18   1   8 164]
#  [ 16   0   4  32]
#  [ 56  52  16  64]]

# img = np.random.randint(0,256,size=[4,4])
# print(img.shape) # 4*4
# img1 = np.array(img)
# print(type(img1))
# img1 = img.view(2,8)  # reshape，resize, view 数组类型进行变换
# print(img1.shape)

# import cv2 
# img = cv2.imread('data/my_PH2/train/masks/0.bmp',0)
# img = np.random.randint(0,10,size=(2,2),dtype=np.uint8)
# img = np.array(img)
# print(img) #(572, 765) (Y,X)
# print(img.shape)
# # cv2.imshow('img',img)
# dst = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# print(dst)
# print(dst.shape)
# dst1 = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
# print(dst1)
# print(dst1.shape)

# [[3 4]
#  [5 8]]

# (2, 2)

# [[3 3 3 4]
#  [3 4 5 5]
#  [4 5 7 7]
#  [5 6 8 9]]

# (4, 4)

# [[3 3 4 4]
#  [3 3 4 4]
#  [5 5 8 8]
#  [5 5 8 8]]

# (4, 4)

# import cv2 
# img = cv2.imread('data/my_PH2/train/images/0.bmp')
# print(img.shape) #(572, 765)
# # h, w = img.shape[0:2]
# # print(img[:,:,0:1])
# img1 = img[:,:,0:1]
# print(img1.shape)



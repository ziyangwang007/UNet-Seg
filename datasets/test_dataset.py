from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image
import cv2

import torchvision.transforms as transforms
class Test_datasets(Dataset):
    def __init__(self, path_Data, config):
        super(Test_datasets, self)
        
        images_list = sorted(os.listdir(path_Data+'test/images/'))
        masks_list = sorted(os.listdir(path_Data+'test/masks/'))
        self.data = []
        for i in range(len(images_list)):
            img_path = path_Data+'test/images/' + images_list[i]
            mask_path = path_Data+'test/masks/' + masks_list[i]
            self.data.append([img_path, mask_path])
        self.transformer = config.test_transformer
        
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        print(img_path)
        # img = np.array(Image.open(img_path).convert('RGB'))
        img_BGR = cv2.imread(img_path, 1)
        img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        img = np.array(img_RGB)
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255
        img, msk = self.transformer((img, msk))
        return img, msk

    def __len__(self):
        return len(self.data)
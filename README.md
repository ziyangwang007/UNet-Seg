<div align="center">
<h1> UNet Zoo for Medical Image Segmentation </h1>

</div>

## Introduction ###

The Exploration of CNN-, ViT-, Mamba-, and KAN-based UNet for Medical Image Segmentation.

Over 10 Segmentation Networks, 7 public benchmark datasets, 6 evaluation metrics are public available!




## Contents ###
- [Segmentation Network List](#Networks)
- [Segmentation Dataset List](#Datasets)
- [Segmentation Metrics List](#Metrics)
- [Environment](#Environment)
- [Usage](#Usage)
- [Reference](#Reference)



## Networks



| Architecture | Name | Reference | Availability |
|:---------------|:----------------|:----------------|:----------------|
| CNN | U-net           | [Ronneberger et al. (2015)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | ✅ |
| CNN | Attention U-net | [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999) |  - [x] |
| CNN | ResUnet         | [Diakogiannis et al. (2020)](https://doi.org/10.1016/j.isprsjprs.2020.01.013) |  - [x] |
| CNN | DenseUnet       | [Li et al. (2020)](https://arxiv.org/pdf/1709.07330) |  [x] |
| CNN | ConvUNeXT       | [Han et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0950705122007572) |  - [x] |
| CNN | TriConvUNeXT    | [Ma et al. (2024)](https://link.springer.com/article/10.1007/s10278-024-01116-8) |  - [-] |
| ViT | TransUNET       | [Chen et al. (2021)](https://arxiv.org/abs/2102.04306) | - [x] |
| ViT | Swin-UNET       | [Hu et al. (2021)](https://arxiv.org/abs/2105.05537) | - [x] |
| ViT | UDTransNet      | [Wang et al. (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0893608024004702) | - [ ] |
| ViT | H2Former        | [He et al. (2023)](https://ieeexplore.ieee.org/document/10093768/) | - [ ] |
| Mamba | Mamba-UNet    | [Wang et al. (2024)](https://arxiv.org/abs/2402.05079) | - [x] |
| Mamba | VM-UNet       | [Ruan et al. (2024)](https://arxiv.org/abs/2402.02491) |  - [ ] |
| KAN | U-KAN           | [Li et al. (2024)](https://arxiv.org/abs/2406.02918) |  - [ ] |
| KAN | KAN-UNet        | [Wang et al. (2024)](https://github.com/ziyangwang007/UNet-Seg) | - [x] |


- CNN-based UNet

UNet, Attention UNet, DenseUNet, ConvUNeXt, 

- ViT-based UNet

TransUNet, SwinUNet

- Mamba-based UNet

Mamba-UNet, VM-UNet

- KAN-based UNet

U-KAN, etc



## Datasets

- [x] Dataset of GLAS  -> [[Official]](https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation/), [[Google Drive]](https://drive.google.com/file/d/1_jtN4XFQ4TC74JiLl07nR1uUr9h5CpQk/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/1SiDo4Dri12bApwpncqEavg) with passcode: 'fp42'
- [x] Dataset of BUSI  -> [[Official]](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/),  [[Google Drive]](https://drive.google.com/file/d/1v4pWl6vQDNmFMJShzhpoxdiQ6Ixvz3gC/view?usp=sharing),  [[Baidu Netdisk]](https://pan.baidu.com/s/14V_lUlJ_XMYMgTYoz851qw) with passcode: '5m5m' 
- [x] Dataset of 2018DSB   -> [[Official]](https://www.kaggle.com/competitions/data-science-bowl-2018/),  [[Google Drive]](https://drive.google.com/file/d/1LnonyBfGWrd4TQqX80WYWxzrO0nYRzXZ/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/15WthemVJWGFPfE4WjRy6xw) with passcode: 'yavx' 
- [x] Dataset of CVC-ClinicDB  -> [[Official]](https://paperswithcode.com/dataset/cvc-clinicdb/),  [[Google Drive]](https://drive.google.com/file/d/1FuZRCkUsKuMezhPKmqP6CrFGL1X4K8RN/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/1f6XNKtOzF2mrFLb_UXKSuQ) with passcode: '3tpy' 
- [x] Dataset of Kvasir-SEG  -> [[Official]](https://arxiv.org/abs/1911.07069/),  [[Google Drive]](https://drive.google.com/file/d/1yFYWp12ZEmbQOqNX--fBJdUc0va_HDq0/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/1JYsAxpKHAXW67m_KnJUUyw) with passcode: '6fgs'
- [x] Dataset of ISIC2016  -> [[Official]](https://challenge.isic-archive.com/landing/2016//), [[Google Drive]](https://drive.google.com/file/d/18RHPlZU_hckZ4STLh7JUhHybMjK7mluC/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/1Bg38aIgwnZwHAqz-qOrLIg) with passcode: 'm2fw' 
- [x] Dataset of PH2  -> [[Official]](https://ieeexplore.ieee.org/document/6610779/),  [[Google Drive]](https://drive.google.com/file/d/1g4byKNeSKzH7qiwbDn3g5FIWWMLUCerX/view?usp=sharing), [[Baidu Netdisk]](https://pan.baidu.com/s/1T_Y4RhJbJ_-J5FM9wLoAXg) with passcode: 'aiax'  

## Metrics

Dice, IoU, Accuracy, Precision, Sensitivity, Specificity


## Environment
* Pytorch
* Some basic python packages: Torchio, Numpy, Scikit-image, SimpleITK, Scipy, Medpy, nibabel, tqdm ......
* For Mamba-related packages, please see [[PyPI (mamba-ssm)]](https://pypi.org/project/mamba-ssm/), [[Official GitHub (mamba)]](https://github.com/state-spaces/mamba), [[PyPI (causal-conv1d)]](https://pypi.org/project/causal-conv1d/) , [[GitHub (causal-conv1d)]](https://github.com/Dao-AILab/causal-conv1d).
* For KAN-related packages, please see [[PyPI (pykan)]](https://pypi.org/project/pykan/), [[Official GitHub (ConvKAN)]](https://github.com/KindXiaoming/pykan), [[PyPI (convkan)]](https://pypi.org/project/convkan/), [[Official GitHub (ConvKAN)]](https://github.com/StarostinV/convkan).


## Usage

1. Download the Code.
```shell
git clone https://github.com/ziyangwang007/UNet-Seg.git 
cd UNet-Seg
```

2. Download the Dataset via Google Drive or Baidu Netdisk to `UNet-Seg/data` folder.


3. Train the model.
```shell
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network UNet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network DenseUnet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network AttU_Net --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network ConvUNeXt --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network SwinUnet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network TransUNet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u train.py --network KANUSeg --datasets 2018DSB
```

4. Test the model.

```shell
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network UNet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network DenseUnet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network AttU_Net --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network ConvUNeXt --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network SwinUnet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network TransUNet --datasets 2018DSB && \

CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets PH2  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets isic16  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets BUSI  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets GLAS  && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets CVC-ClinicDB && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets Kvasir-SEG && \
CUDA_VISIBLE_DEVICES=0 python -u test.py --network KANUSeg --datasets 2018DSB

```

## Reference
```bibtex
TBC

TBC

TBC
```
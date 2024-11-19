import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from datasets.test_dataset import Test_datasets
from tensorboardX import SummaryWriter
from models.vision_mamba import MambaUnet
from models.ConvKANeXt import ConvKANeXt as KANUSeg
from models.UNet import UNet
from models.ATTUNet import AttU_Net
from models.DenseUnet import Dense_Unet as DenseUnet
from models.SwinUnet import SwinUnet, SwinUnet_config
from models.TransUnet import get_transNet as TransUNet
from models.ConvUNext import ConvUNeXt
# from models.mamba_sys import VSSM
import argparse
from engine import *
import os
import sys
from config import get_config
from utils import *
# from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")




parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str,
                    default='ConvUNeXt', help='choose the model. UNet, DenseUnet, KANUSeg, AttU_Net, ConvUNeXt, mamba_UNet, SwinUnet, TransUNet.')
parser.add_argument('--datasets', type=str,
                    default='PH2', help='choose the dataset. PH2, isic16, BUSI, GLAS, CVC-ClinicDB, Kvasir-SEG, 2018DSB.')
args = parser.parse_args()


# Save parsed arguments into a separate file for easy access
with open('parsed_args.txt', 'w') as f:
    f.write(f"network={args.network}\n")
    f.write(f"datasets={args.datasets}\n")



# Ensure the file is written and available
while not os.path.isfile('parsed_args.txt'):
    time.sleep(10)

print('#----------Init----------#')




from config import get_config
from configs.config_setting import setting_config




config = setting_config

sys.path.append(config.work_dir + '/')
log_dir = os.path.join(config.work_dir, 'log')
checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
resume_model = os.path.join(checkpoint_dir, 'latest.pth')
outputs = os.path.join(config.work_dir, 'outputs')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
if not os.path.exists(outputs):
    os.makedirs(outputs)

global logger
logger = get_logger('train', log_dir)
global writer
writer = SummaryWriter(config.work_dir + 'summary')

log_config_info(config, logger)


print('#----------GPU init----------#')
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
set_seed(config.seed)
torch.cuda.empty_cache()





print('#----------Preparing dataset----------#')
test_dataset = Test_datasets(config.data_path, config)

test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=True)

print('#----------Prepareing Model----------#')

if config.network == 'mamba_UNet':
    model_cfg = config.model_config
    model = MambaUnet(model_cfg,128,1)
    model.load_from(model_cfg)
elif config.network == 'SwinUnet':
    model_cfg = SwinUnet_config()
    model = SwinUnet(model_cfg,img_size=224,num_classes=1)
elif config.network == 'KANUSeg':
    model = KANUSeg(3,1)
elif config.network == 'ConvUNeXt':
    model = ConvUNeXt(3,1)
elif config.network == 'TransUNet':
    model = TransUNet(1)
elif config.network == 'UNet':
    model = UNet(3,1)       
elif config.network == 'DenseUnet':
    model = DenseUnet(3,1)  
elif config.network == 'AttU_Net':
    model = AttU_Net(3,1).to('cuda')  
else: raise Exception('network in not right!')
model = model.cuda()

cal_params_flops(model, 224, logger)
print('#----------Testing----------#')
best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
model.load_state_dict(best_weight)
model.eval()
preds = []
gts = []
loss_list = []
with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader)):
        img, msk = data
        img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
        out = model(img)
        msk = msk.squeeze(1).cpu().detach().numpy()
        gts.append(msk)
        if type(out) is tuple:
            out = out[0]
        out = out.squeeze(1).cpu().detach().numpy()
        preds.append(out) 
        if i % 1 == 0:
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=None)
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds>=config.threshold, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    precision = float(TP) / float(TP+FP) if float(TP+FP) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    # if test_data_name is not None:
    #     log_info = f'test_datasets_name: {test_data_name}'
    #     print(log_info)
    #     logger.info(log_info)
    # log_info = f'test of best model, f1_or_dsc: {f1_or_dsc}, miou: {miou},  accuracy: {accuracy},\
    #         precision:{precision}, sensitivity: {sensitivity}, specificity: {specificity},  confusion_matrix: {confusion}'
    log_info = (
    f'test of best model, '
    f' {f1_or_dsc:.4f} & '
    f' {miou:.4f} & '
    f' {accuracy:.4f} & '
    f' {precision:.4f} & '
    f' {sensitivity:.4f} & '
    f'{specificity:.4f}, '
)
    print(log_info)
    logger.info(log_info)   

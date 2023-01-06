import numpy as np
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import helps
from model import Model
from data.cudl import CustomDataset
import data.common

mps_device = torch.device("cuda:0")
doc_dir = "../../../data/publaynet/"
result_dir = "../../../results/transfer/"

helps.mkdir(result_dir)

#(amp=False, augment=True, batch_size=16, betas=(0.9, 0.999), blur_key='blur_gamma', data_root='/home/varun/Research/dataset', data_test='', data_train='', data_val='', dataset=None, demo=True, demo_input_dir='/home/varun/Research/dataset/GOPRO_Large/test/GOPR0384_11_00/blur_gamma', demo_output_dir='', device=device(type='cuda', index=0), device_index=0, device_type='cuda', dist_backend='nccl', distributed=False, do_test=False, do_train=False, do_validate=False, downsample='Gaussian', dtype=torch.float32, dtype_eval=torch.float32, end_epoch=1000, epsilon=1e-08, gamma=0.5, gaussian_pyramid=True, init_method='env://', init_scale=1024.0, kernel_size=5, launched=False, load_epoch=1000, loss='1*L1', lr=0.0001, master_addr='127.0.0.1', master_port='8023', metric='PSNR,SSIM', milestones=[500, 750, 900], model='MSResNet', momentum=0.9, n_GPUs=1, n_feats=64, n_resblocks=19, n_scales=3, num_workers=7, optimizer='ADAM', patch_size=256, precision='single', pretrained='', rank=0, rgb_range=255, save_dir='../experiment/GOPRO_L1', save_every=10, save_results='all', scheduler='step', seed=1670040074, split_batch=1, start_epoch=1001, stay=False, template='', test_every=10, validate_every=10, weight_decay=0, world_size=1)
args = None

args_path = "../experiment/GOPRO_L1/args.pt"
weights_ = "/home/kaundinya/results/transfer/models/model_best.pth"
args = torch.load(args_path)
print(mps_device)
args.__setattr__('device_type' , mps_device)
args.__setattr__('dtype' , torch.float32)
args.__setattr__('dtype_eval' , torch.float32)
args.__setattr__('device' , torch.device("cuda:0"))
args.__setattr__('demo', False)

print(args)

model = Model(args)
helps.load_checkpoint(model, weights_)

model.to(mps_device)

model.eval()

cdata = CustomDataset(document_dir= doc_dir,interpolation=TF.InterpolationMode.BICUBIC, gaussian_pyramid=True, output_dim=(512,512),filter=False, type='val')

dataloader = DataLoader(dataset=cdata, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
print("length of the dataset", len(dataloader.dataset))
original_dir = os.path.join(result_dir,"original")
blurred_dir = os.path.join(result_dir,"blurred")
restored_dir = os.path.join(result_dir,"restored")


with torch.no_grad():
    psnr_val_rgb = []
    topil = TF.ToPILImage()
    for ii, data_test in enumerate(tqdm(dataloader), 0):
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()
        blurredImageSet = data_test[0]
        sharpImageSet = data_test[1]
        #blurr = blurredImageSet.to(mps_device)
        #gt = data_test[1].cpu().detach()
        filenames = data_test[2]

        input, target =  data.common.to(blurredImageSet, sharpImageSet, device=mps_device, dtype=torch.float32)
        restored = model(input)
        restored = torch.clamp(restored[0],0,1)

        restored = restored.cpu().detach()
        for batch in range(len(restored)):
            restored_img = restored[batch]
            psnr_val_rgb.append(helps.torchPSNR(restored_img, sharpImageSet[batch][0]))
            helps.save_img((os.path.join(restored_dir, filenames[batch]+'.png')), topil(restored_img))
            helps.save_img((os.path.join(original_dir, filenames[batch]+'.png')), topil(sharpImageSet[batch][0]))
            helps.save_img((os.path.join(blurred_dir, filenames[batch]+'.png')), topil(blurredImageSet[batch][0]))
            print(sharpImageSet[batch][0][0].shape,"tree")
            plt.ion()
            someplt = plt.imshow(blurredImageSet[batch][0][0], cmap='Greys_r')
            someplt1 = plt.imshow(blurredImageSet[batch][0][1], cmap='Greys_r')
            someplt2 = plt.imshow(blurredImageSet[batch][0][2], cmap='Greys_r')
            
        if(ii != 0 and ii % 50 == 0):
            break
    
    print("PSNR average", sum(psnr_val_rgb)/len(psnr_val_rgb))
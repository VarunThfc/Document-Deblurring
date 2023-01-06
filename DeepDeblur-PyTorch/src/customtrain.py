import numpy as np
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import time

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import helps
from model import Model
from data.cudl import CustomDataset
from loss import Loss
from optim import Optimizer

import data.common

mps_device = torch.device("cuda:0")
doc_dir = "../../../data/publaynet/"
result_dir = "../../../results/transfer"

helps.mkdir(result_dir)


args_path = "../experiment/GOPRO_L1/args.pt"
weights_ = "../experiment/GOPRO_L1/models/model-1000.pt"
optim_ = "../experiment/GOPRO_L1/optim/optim-1000.pt"
loss_ = "../experiment/GOPRO_L1/loss.pt"
args = torch.load(args_path)
print(mps_device)
args.__setattr__('device_type' , mps_device)
args.__setattr__('dtype' , torch.float32)
args.__setattr__('dtype_eval' , torch.float32)
args.__setattr__('device' , torch.device("cuda:0"))
args.__setattr__('demo', False)
args.__setattr__('amp', True)

print(args)


start_epoch = 1
mode = "Deblurring"
session = "Deblur"

result_dir   = os.path.join(result_dir)
original_dir = os.path.join(result_dir, "original")
blurred_dir  = os.path.join(result_dir, "blurred")
restored_dir = os.path.join(result_dir, "restored")
model_dir    = os.path.join(result_dir, "models")
log_dir      = os.path.join(result_dir, "logs")


helps.mkdir(result_dir)
helps.mkdir(original_dir)
helps.mkdir(blurred_dir)
helps.mkdir(restored_dir)
helps.mkdir(model_dir)

writer = SummaryWriter(log_dir)

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

##LOAD everything

model_restoration = Model(args)
model_restoration.cuda()

helps.load_checkpoint(model_restoration, weights_)
model_restoration.parallelize()
optimizer = Optimizer(args, model_restoration)
# if os.path.exists(self.save_name):
#             state = torch.load(self.save_name, map_location=self.args.device)

#             self.loss_stat = state['loss_stat']
#             if 'metric_stat' in state:
#                 self.metric_stat = state['metric_stat']
#             else:
#                 pass

criterion = Loss(args, model=model_restoration, optimizer=optimizer)

###DATA AND LOADER

cdata = CustomDataset(document_dir= doc_dir,interpolation=TF.InterpolationMode.BICUBIC, gaussian_pyramid=True, output_dim=(256,256),filter=False)
dataloader = DataLoader(dataset=cdata, batch_size=4, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)
len_dataset = len(cdata)
print(len_dataset, "trian")
val_data = CustomDataset(document_dir= doc_dir,interpolation=TF.InterpolationMode.BICUBIC, gaussian_pyramid=True, output_dim=(256,256),filter=False,type="val")
val_loader = DataLoader(dataset=val_data, batch_size=4, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
len_valdataset = len(val_data)

print('bale')
# %%
start_epoch = 1
num_samples_per_epoch = 1000
num_val_samples = 400
data_epochs = len_dataset//num_samples_per_epoch
actual_epochs = 10
num_epochs = actual_epochs*data_epochs
num_tb_samples = 100
tb_epochs = len_dataset//num_tb_samples

# %%
###TRAIN
model_restoration.train()
model_restoration.to(mps_device)
criterion.train()
torch.set_grad_enabled(True)

scaler = amp.GradScaler(
            init_scale=args.init_scale,
            enabled=args.amp
        )

best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, actual_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    model_restoration.train()
    criterion.train()
    for i, data_train in enumerate(tqdm(dataloader), start=1):
        optimizer.zero_grad()
        blurredImageSet = data_train[0]
        sharpImageSet = data_train[1]
        input, target =  data.common.to(blurredImageSet, sharpImageSet, device=mps_device, dtype=torch.float32)
        with amp.autocast(args.amp):
            output = model_restoration(input)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer.G)
        scaler.update()
        criterion.normalize()
        criterion.step()
        optimizer.schedule(criterion.get_last_loss())
        if i%num_tb_samples == 0:
            writer.add_scalar("data/training_loss", loss.item(), (epoch-1) * tb_epochs + (i//num_tb_samples))
            
        ##EVALUATE
        if i%num_samples_per_epoch == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), start=1):
                blurredImageSetVal = data_val[0]
                sharpImageSetVal = data_val[1]
                inputVal, targetVal =  data.common.to(blurredImageSetVal, sharpImageSetVal, device=mps_device, dtype=torch.float32)

                with torch.no_grad():
                    restored = model_restoration(inputVal)
                restored = restored[0]

                for res,tar in zip(restored,targetVal):
                    psnr_val_rgb.append(helps.torchPSNR(res, tar))
                
                if ii%num_val_samples==0:
                    break
            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()
            
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = (epoch-1)*data_epochs+(i//num_samples_per_epoch)
                torch.save({'epoch': best_epoch, 
                            'state_dict': model_restoration.state_dict(),
                            'optimizer' : optimizer.state_dict()
                            }, os.path.join(model_dir,"model_best.pth"))
            current_epoch=(epoch-1)*data_epochs+(i//1000)
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (current_epoch, psnr_val_rgb, best_epoch, best_psnr))
            
            writer.add_scalar("data/validation_psnr", psnr_val_rgb, current_epoch)
            
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time, epoch_loss))
            print("------------------------------------------------------------------")

            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_latest.pth"))

            epoch_start_time = time.time()
            epoch_loss = 0
            train_id = 1
            model_restoration.train()
        
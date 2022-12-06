# %%
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from MPRNet import MPRNet
import utils
from customDataSet import CustomDataset
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from pdb import set_trace as stx

cuda_device = torch.device("cuda")
doc_dir = "../../publaynet/"
result_dir = "../results/transfer"
utils.mkdir(result_dir)

# %%
mode = "Deblurring"
session = "MPRNet"

result_dir   = os.path.join(result_dir)
original_dir = os.path.join(result_dir, "original")
blurred_dir  = os.path.join(result_dir, "blurred")
restored_dir = os.path.join(result_dir, "restored")
model_dir    = os.path.join(result_dir, "models")
log_dir      = os.path.join(result_dir, "logs")

utils.mkdir(result_dir)
utils.mkdir(original_dir)
utils.mkdir(blurred_dir)
utils.mkdir(restored_dir)
utils.mkdir(model_dir)
utils.mkdir(log_dir)

writer = SummaryWriter(log_dir)

# %%
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

# %%
model_restoration = MPRNet()
model_restoration.cuda()
if len(device_ids)>1:
    model_restoration = nn.DataParallel(model_restoration, device_ids = device_ids)

# %%
weights_ = "./pretrained_models/model_deblurring.pth"
utils.load_checkpoint(model_restoration, weights_)

# %%
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

# %%
train_data = CustomDataset(doc_dir, output_dim=(256,256), filter=False, type="train")
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
len_dataset = len(train_data)

# %%
val_data = CustomDataset(doc_dir, output_dim=(256,256), filter=False, type="val")
val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=True, num_workers=2, drop_last=False, pin_memory=True)
len_valdataset = len(val_data)

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
new_lr = 2e-4
optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)

# %%
warmup_epochs = 5
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs-warmup_epochs, eta_min=1e-6)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

# %%
best_psnr = 0
best_epoch = 0

for epoch in range(start_epoch, actual_epochs + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), start=1):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0].cuda()
        input_ = data[1].cuda()

        restored = model_restoration(input_)
 
        # Compute loss at each stage
        loss_char = np.sum([criterion_char(restored[j],target) for j in range(len(restored))])
        loss_edge = np.sum([criterion_edge(restored[j],target) for j in range(len(restored))])
        loss = (loss_char) + (0.05*loss_edge)
       
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
        if i%num_tb_samples == 0:
            writer.add_scalar("data/training_loss", loss.val, (epoch-1) * tb_epochs + (i//num_tb_samples))

        #### Evaluation ####
        if i%num_samples_per_epoch == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate((val_loader), start=1):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0]

                for res,tar in zip(restored,target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                
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

            # torch.save({'epoch': epoch, 
            #             'state_dict': model_restoration.state_dict(),
            #             'optimizer' : optimizer.state_dict()
            #             }, os.path.join(model_dir,f"model_epoch_{epoch}.pth")) 

            writer.add_scalar("data/learning_rate", scheduler.get_last_lr()[0], current_epoch)
            writer.add_scalar("data/validation_psnr", psnr_val_rgb, current_epoch)
            scheduler.step()
            
            print("------------------------------------------------------------------")
            print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
            print("------------------------------------------------------------------")

            torch.save({'epoch': epoch, 
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,"model_latest.pth"))

            epoch_start_time = time.time()
            epoch_loss = 0
            train_id = 1
            model_restoration.train()
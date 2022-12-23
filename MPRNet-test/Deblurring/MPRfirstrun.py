# %%
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF

import numpy as np
import pandas as pd
import pickle

from MPRNet import MPRNet
import utils

gpu_device = torch.device("cuda")
doc_dir = "../../publaynet/"
result_dir = "../results/"
utils.mkdir(result_dir)

# %%
model_restoration = MPRNet()

# %%
weights_ = "./pretrained_models/model_deblurring.pth"
utils.load_checkpoint(model_restoration, weights_)

# %%
model_restoration.to(gpu_device)
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# %%
from customDataSet import CustomDataset

# %%
data = CustomDataset(doc_dir,(400,400),False)
dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
original_dir = os.path.join(result_dir,"original")
blurred_dir = os.path.join(result_dir,"blurred")
restored_dir = os.path.join(result_dir,"restored")

# %%
with torch.no_grad():
    psnr_val_rgb = {}
    topil = TF.ToPILImage()
    for ii, data_test in enumerate(tqdm(dataloader), 0):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        blurr = data_test[0].cuda()
        gt = data_test[1].cpu().detach()
        filenames = data_test[2]

        restored = model_restoration(blurr)
        restored = torch.clamp(restored[0],0,1)

        restored = restored.cpu().detach()
        blurr = blurr.cpu().detach()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            psnr_val_rgb[filenames[batch]] = float(utils.torchPSNR(restored_img, gt[batch]).cpu().detach().numpy())
            utils.save_img((os.path.join(restored_dir, filenames[batch]+'.png')), topil(restored_img))
            utils.save_img((os.path.join(original_dir, filenames[batch]+'.png')), topil(gt[batch]))
            utils.save_img((os.path.join(blurred_dir, filenames[batch]+'.png')), topil(blurr[batch]))

# %%
with open("PSNRs.pkl","wb") as f:
    pickle.dump(psnr_val_rgb,f)



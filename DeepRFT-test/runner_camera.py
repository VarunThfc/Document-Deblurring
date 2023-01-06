import os
import pickle
from tqdm import tqdm

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.transforms as TF

import utils
from layers import *
from DeepRFT_MIMO import DeepRFT as mynet
from get_parameter_number import get_parameter_number

from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss

from cameraDataSet import CameraDataset

# %%

doc_dir = "./camera/"
result_dir = "./results/camera"
weights = "./results/transfer/models/model_best.pth"

num_res = 8         # num of resblocks, [Small, Med, PLus]=[4, 8, 20]
win = 256           # window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512
gpus = 0            # CUDA_VISIBLE_DEVICES

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)

# %%

# model_restoration = mynet()
model_restoration = mynet(num_res=num_res, inference=True)
# print number of model
get_parameter_number(model_restoration)
# utils.load_checkpoint(model_restoration, weights)
utils.load_checkpoint_compress_doconv(model_restoration, weights)
print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# %%

data = CameraDataset(doc_dir,(400,400),False)
dataloader = DataLoader(dataset=data, batch_size=1, shuffle=False, drop_last=False, pin_memory=True)

# %%

input_dir = os.path.join(result_dir,"input")
output_dir = os.path.join(result_dir,"output")

utils.mkdir(result_dir)
utils.mkdir(input_dir)
utils.mkdir(output_dir)

# %%

with torch.no_grad():
    topil = TF.ToPILImage()
    for ii, data_test in enumerate(tqdm(dataloader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        input_      = data_test[0].cuda()
        filenames   = data_test[1]

        print(input_.shape, len(filenames))   # torch.Size([1, 3, 400, 400]) torch.Size([1, 3, 400, 400]) 1
        
        _, _, Hx, Wx = input_.shape
        input_re, batch_list = window_partitionx(input_, win)
        restored = model_restoration(input_re)
        restored = window_reversex(restored, win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored = restored.cpu().detach().numpy()

        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored_img)
            restored_img = torch.tensor(restored_img)

            blurred_img = input_[batch]
            utils.save_img((os.path.join(output_dir, filenames[batch]+'.png')), topil(restored_img))
            utils.save_img((os.path.join(input_dir, filenames[batch]+'.png')), topil(blurred_img))

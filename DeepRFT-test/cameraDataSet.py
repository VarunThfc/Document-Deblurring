import glob
import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset
import torchvision.transforms.transforms as TF

import random

class CameraDataset(Dataset):
    
    def __init__(self, document_dir, output_dim=(400,400),filter=True, interpolation=TF.InterpolationMode.BICUBIC):
        super(CameraDataset, self).__init__()

        self.data = glob.glob(os.path.join(document_dir,"*.jpg")) ##Contents inside Path
        print('size', len(self.data))

        self.img_dim = output_dim
        self.transforms = TF.RandomResizedCrop(size=self.img_dim,scale=(0.1,0.2),ratio=(0.5,2),interpolation=interpolation)
        
        self.totensor = TF.ToTensor()
    
    def __len__(self):   ##Mandatory override criteria
        return len(self.data)       
    
    def __getitem__(self, idx):
        img_path = self.data[idx]   
        img = Image.open(img_path)
        img = self.transforms(img)
        img = self.totensor(img)
        return img, os.path.splitext(os.path.split(img_path)[-1])[0]

        
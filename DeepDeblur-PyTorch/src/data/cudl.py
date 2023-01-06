import glob
import cv2
from PIL import Image
import os
from data import common
import numpy as np
import pandas as pd
from scipy.ndimage import rotate

import torch
from torch.utils.data import Dataset
import torchvision.transforms.transforms as TF

import json
import random

def filtered(filename):
    data = json.load(filename)
    validFileIds = set();
    for annotation in data['annotations']:
        if(annotation['category_id'] == 1):
            validFileIds.add(annotation['image_id'])
    validFileNames = set();
    for images in data['images']:
        if(images['id'] in validFileIds):
            validFileNames.add(images['file_name'])
    print("fileName", len(validFileNames))
    return validFileNames

def motion_blur(img, ksize):
    kernel = np.zeros((ksize, ksize))
    kernel[ksize//2, :] = 1
    kernel = kernel*np.random.rand(ksize,ksize)
    kernel = kernel / np.sum(kernel)
    random_angle = random.randint(0, 180)
    kernel = rotate(kernel, random_angle)
    return cv2.filter2D(img, -1, kernel)

def random_blur(img_to_blur, list_of_blurs,p=None):
    pickblur = TF.RandomChoice(list_of_blurs,p=p)
    for _ in range(np.random.randint(1,3)):
        ksize = np.random.randint(2,4)
        img_to_blur = pickblur(img_to_blur,2*ksize-1)
    return img_to_blur

class CustomDataset(Dataset):
    
    def __init__(self, document_dir,  interpolation=TF.InterpolationMode.BICUBIC, gaussian_pyramid=True, output_dim=(400,400),filter=True,type="train"):
        super(CustomDataset, self).__init__()
        self.n_scales = 3
        self.gaussian_pyramid = gaussian_pyramid
        self.data = glob.glob(os.path.join(document_dir,type,"*.jpg")) ##Contents inside Path
        train_json = open(document_dir + "/labels/" + "train.json")
        if filter:
            filteredFileSet = filtered(train_json)
            self.data = [file  for file in self.data  if file.split('/')[-1] in filteredFileSet] ##Check this once we have the data
        print('size', len(self.data))

        self.img_dim = output_dim

        self.transforms = TF.Compose([TF.RandomVerticalFlip(.3),
                                      TF.RandomHorizontalFlip(.3),
                                      TF.RandomRotation(45,fill=255,interpolation=interpolation),
                                      TF.RandomResizedCrop(size=self.img_dim,scale=(0.1,0.2),ratio=(0.2,2),interpolation=interpolation)
                                    ])
        
        self.totensor = TF.ToTensor()
        
        self.listOfBlurs = [
                        lambda img, ksize : cv2.GaussianBlur(img,(ksize,ksize),cv2.BORDER_DEFAULT),
                        lambda img, ksize : cv2.blur(img,(ksize,ksize)),
                        lambda img, ksize : motion_blur(img, 4*ksize+np.random.choice([1,-1]))
                    ]## Resize dimension experiment
        self.p = [1,1,10]
    
    def __len__(self):   ##Mandatory override criteria
        return len(self.data)       

    
    def __getitem__(self, idx):
        img_path = self.data[idx]   
        img = Image.open(img_path)
        img = self.transforms(img)
        blurimg = Image.fromarray(random_blur(np.asarray(img),self.listOfBlurs, self.p))
        img = self.totensor(img).permute((1,2,0))
        blurimg = self.totensor(blurimg).permute((1,2,0))
        img = img.numpy()
        blurimg = blurimg.numpy()
        listOfImages = [blurimg, img]
        #print('before pyramid', img.shape, blurimg.shape)

        if self.gaussian_pyramid:
            listOfImages = common.generate_pyramid(*listOfImages, n_scales=self.n_scales)
        listOfImages = common.np2tensor(listOfImages)
        blur = listOfImages[0][0];
        sharp = listOfImages[0][1];
        return blur, sharp, os.path.splitext(os.path.split(img_path)[-1])[0]

        
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json


class CustomDataset(Dataset):
    def __init__(self):
        self.imgs_path = "./publaynet/" ##Base path
        file_list = glob.glob(self.imgs_path + "*") ##Contents inside Path
        train_json = open('./publaynet/train.json')
        filteredFileSet = self.filter(train_json)
        self.data = []
        for class_path in file_list:
            for img_path in glob.glob(class_path + "/*.jpg"): ##Loop over all the images
                if(img_path.split("/")[-1] in filteredFileSet):
                    self.data.append(img_path) ##Check this once we have the data
        print('size', len(self.data))
        #self.class_map = {"dogs" : 0, "cats": 1}
        self.img_dim = (416, 416) ## Reszie dimension experiment
    
    def __len__(self):   ##Mandatory override criteria
        return len(self.data)
    
    def filter(self, filename):
        data = json.load(filename)
        count = 0;
        validFileIds = set();
        for annotation in data['annotations']:
            if(annotation['category_id'] == 1):
                count = count+1
                validFileIds.add(annotation['image_id'])
        print("filtered", len(validFileIds))
        count = 0;
        validFileNames = set();
        for images in data['images']:
            if(images['id'] in validFileIds):
                count = count+1
                validFileNames.add(images['file_name'])
        print("fileName", len(validFileNames))
        return validFileNames
        
    def __getitem__(self, idx):
        img_path = self.data[idx]   ##THink of modding if required
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim) ##Should I just return or resize and takecare in transforms
        #class_id = self.class_map[class_name] ##Is this required ??
        img_tensor = torch.from_numpy(img) ##Convert to tensore
        img_tensor = img_tensor.permute(2, 0, 1) ##SWAP channel to adhere to torch
        #class_id = torch.tensor([class_id])
        return img_tensor
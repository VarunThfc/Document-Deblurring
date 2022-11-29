import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import orjson as json
import random
from scipy.ndimage import rotate

class CustomDataset(Dataset):
    
    def __init__(self):

        self.imgs_path = "./publaynet/train" ##Base path
        file_list = glob.glob(self.imgs_path + "*") ##Contents inside Path
        train_json = './publaynet/labels/train.json'
        filteredFileSet = self.filter(train_json)
        self.data = []
        for class_path in file_list:
            for img_path in glob.glob(class_path + "/*.jpg"): ##Loop over all the images
                if(img_path.split("/")[-1] in filteredFileSet):
                    self.data.append(img_path) ##Check this once we have the data
        print('size', len(self.data))
        #self.class_map = {"dogs" : 0, "cats": 1}
        self.img_dim = (400, 400)
        self.listOfTransformations = [
                        lambda img, ksize : cv2.GaussianBlur(img,(ksize,ksize),cv2.BORDER_DEFAULT), 
                        lambda img, ksize : cv2.medianBlur(img,ksize if ksize<=3 else 3), 
                        lambda img, ksize : cv2.blur(img,(ksize,ksize)),
                        lambda img, ksize : self.motion_blur(img, 2*ksize+1)
                    ]## Reszie dimension experiment
    
    def __len__(self):   ##Mandatory override criteria
        return len(self.data)
    
    def filter(self, filename):
        with open(filename,"rb") as f:
            data = json.loads(f.read())
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

    def motion_blur(self, img, ksize):
        kernel = np.zeros((ksize, ksize))
        kernel[ksize//2, :] = 1
        kernel = kernel / np.sum(kernel)
        random_angle = random.randint(0, 180)
        kernel = rotate(kernel, random_angle)
        return cv2.filter2D(img, -1, kernel)
    
    def random_blur(self, img_to_blur):
        num = random.randint(1, len(self.listOfTransformations)+1)
        for transformation in np.random.choice(self.listOfTransformations, num, replace=True):
            ksize = np.random.randint(1,3)
            img_to_blur = transformation(img_to_blur,2*ksize-1)
        return img_to_blur
    
    def totensor(self, img):
        assert isinstance(img, np.ndarray)
        # convert image to tensor
        assert (img.ndim > 1) and (img.ndim <= 3)
        if img.ndim == 2:
            img = img[:, :, None]
            tensor_img = torch.from_numpy(np.ascontiguousarray(
            img.transpose((2, 0, 1))))
        if img.ndim == 3:
            tensor_img = torch.from_numpy(np.ascontiguousarray(
            img.transpose((2, 0, 1))))
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255.0)
        else:
            return tensor_img        
    
    def __getitem__(self, idx):
        img_path = self.data[idx]   
        img = cv2.imread(img_path)
        (h,w,_) = img.shape
        hcrop,wcrop = np.random.randint(h-self.img_dim[0]), np.random.randint(w-self.img_dim[1])
        cropimg = img[hcrop:hcrop+self.img_dim[0],wcrop:wcrop+self.img_dim[1]]
        img_to_blur = cropimg.copy()
        # img = cv2.resize(img, self.img_dim) ##Should I just return or resize and takecare in transforms
        # img_to_blur = cv2.resize(img, self.img_dim)
        img_to_blur = self.random_blur(img_to_blur)
        #class_id = torch.tensor([class_id])
        img_to_blur = self.totensor(img_to_blur)
        img = self.totensor(cropimg)
        return img_to_blur, img
        
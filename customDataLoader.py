from customDataSet import CustomDataset
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


dataset = CustomDataset()
print(dataset)

data_loader = DataLoader(dataset, batch_size=300, shuffle=True)

for imgs in data_loader:
    break
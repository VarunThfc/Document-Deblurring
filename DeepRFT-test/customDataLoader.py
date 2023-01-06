from customDataSet import CustomDataset
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import datetime

def main():
    dataset = CustomDataset()
    print(dataset)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers = 4)
    now = datetime.datetime.now()
    for imgs in data_loader:
        plt.imshow(imgs[0][0].permute(1, 2, 0))
        print(imgs[0][0].permute(1, 2, 0).shape);
        print(imgs[1][0].permute(1, 2, 0).shape);

        plt.show()
        plt.imshow(imgs[1][0].permute(1, 2, 0))
        plt.show()
        continue
    then = datetime.datetime.now();
    delta = then - now
    print(delta, delta.total_seconds())


if __name__ == "__main__":
    main()
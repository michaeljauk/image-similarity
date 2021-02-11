import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class TotallyLooksLikeDataset(Dataset):
    def __init__(self, root_dir, setSize, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.setSize = setSize

        self.leftfiles = os.listdir(root_dir + '/left')
        self.rightfiles = os.listdir(root_dir + '/right')

    def __len__(self):
        return self.setSize

    def __getitem__(self, idx):
        # Train with 50% similar, 50% dissimilar:
        # img_index = int(idx / 2) # we dont want to jump over every other similar pair, do we?
        # if idx % 2 == 0: # similar pair
        #     img1 = Image.open(self.root_dir + '/left/' + self.leftfiles[img_index])
        #     img2 = Image.open(self.root_dir + '/right/' + self.rightfiles[img_index])
        #     label = 1.0
        # else: # dissimilar pair
        #     img1 = Image.open(self.root_dir + '/left/' + self.leftfiles[img_index])
        #     if img_index == 0: # choose image above or below, we just dont want to choose the similar one
        #         img_index += 1
        #     else:
        #         img_index -= 1
        #     img2 = Image.open(self.root_dir + '/right/' + self.rightfiles[img_index])
        #     label = 0.0

        # Train only with similar images:
        img1 = Image.open(self.root_dir + '/left/' + self.leftfiles[idx])
        img2 = Image.open(self.root_dir + '/right/' + self.rightfiles[idx])
        label = 1.0

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))

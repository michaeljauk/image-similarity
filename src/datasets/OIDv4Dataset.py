import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import PIL.ImageOps
import random
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import os
import config

# https://github.com/EscVM/OIDv4_ToolKit'
# https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html
# https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv


class OIDv4Dataset(Dataset):

    def __init__(self, root_dir, toolkit_dir, sub, classes, type_csv, limit, transform=None, should_invert=False):
        self.download(root_dir, toolkit_dir, sub, classes, type_csv, limit)

        self.imageFolderDataset = datasets.ImageFolder(
            root=f'{root_dir}/{type_csv}')
        # can be accomplished by self.imageFolderDataset.classes
        # self.categories = [directory for directory in os.listdir(root_dir)]
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        # TODO: maybe cut out squares of images before transforming them

        # tuple consists of imagepath and index of category/class
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        same_class = random.randint(0, 1)
        if same_class:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # Loop until another image of the same class has been found
                if img0_tuple[0] != img1_tuple[0] and img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                # Loop until another image of another class has been found
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0]).convert("RGB")
        img1 = Image.open(img1_tuple[0]).convert("RGB")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def download(self, root_dir, toolkit_dir, sub, classes, type_csv, limit):
        os.system(
            f'python "{toolkit_dir}/main.py" downloader_ill --sub {sub} --classes {classes} --type_csv {type_csv} --limit {limit} --Dataset "{root_dir}"')

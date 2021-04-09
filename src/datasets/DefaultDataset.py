import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps
import random
import numpy as np

import torchvision
from torch.utils.data import DataLoader


class DefaultDataset(Dataset):

    def __init__(self, root_dir, transform=None, should_invert=False):
        self.imageFolderDataset = datasets.ImageFolder(root=root_dir)
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

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# We can scrap that, right?


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def main():
    root_dir = "D:/datasets/256_ObjectCategories/"

    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = Caltech256Dataset(
        root_dir=root_dir, transform=transformations, should_invert=False)

    vis_dataloader = DataLoader(dataset,
                                shuffle=True,
                                num_workers=2,
                                batch_size=8)
    dataiter = iter(vis_dataloader)

    example_batch = next(dataiter)
    concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(example_batch[2].numpy())


if __name__ == "__main__":
    main()

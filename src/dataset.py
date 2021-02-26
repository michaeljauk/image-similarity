import torchvision.transforms as transforms

from .datasets.Caltech256Dataset import Caltech256Dataset
from .datasets.TotallyLooksLikeDataset import TotallyLooksLikeDataset

import config

default_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_dataset(dataset_name, transformations=default_transformations, should_invert=False):
    if dataset_name == 'caltech_256':
        return Caltech256Dataset(
            root_dir=config.dataset_paths[dataset_name],
            transform=transformations,
            should_invert=should_invert)

    if dataset_name == 'totally_looks_like':
        return TotallyLooksLikeDataset(
            root_dir=config.dataset_paths[dataset_name],
            transform=transformations)

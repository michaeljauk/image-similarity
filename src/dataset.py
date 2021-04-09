import torchvision.transforms as transforms

from .datasets.DefaultDataset import DefaultDataset

import config

default_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_dataset(dataset_name, transformations=default_transformations, should_invert=False, dataset_config=None):
    if dataset_name == 'default':
        return DefaultDataset(
            root_dir=config.datasets[dataset_name],
            transform=transformations,
            should_invert=should_invert)

    if dataset_name == 'oid':
        return DefaultDataset(
            root_dir=config.datasets[dataset_name] + "/train",
            transform=transformations,
            should_invert=should_invert)

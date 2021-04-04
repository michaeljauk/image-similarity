import torchvision.transforms as transforms

from .datasets.DefaultDataset import DefaultDataset
from .datasets.TotallyLooksLikeDataset import TotallyLooksLikeDataset
from .datasets.OIDv4Dataset import OIDv4Dataset

import config

default_transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def get_dataset(dataset_name, transformations=default_transformations, should_invert=False, dataset_config=None):
    if dataset_name == 'default':
        return DefaultDataset(
            root_dir=config.dataset_paths[dataset_name],
            transform=transformations,
            should_invert=should_invert)

    if dataset_name == 'totally_looks_like':
        return TotallyLooksLikeDataset(
            root_dir=config.dataset_paths[dataset_name],
            transform=transformations)

    if dataset_name == 'oidv4':
        return OIDv4Dataset(
            root_dir=config.datasets["oidv4"],
            toolkit_dir=config.tools["oidv4_toolkit"],
            sub=dataset_config["sub"],
            classes=dataset_config["classes"],
            type_csv=dataset_config["type_csv"],
            limit=dataset_config["limit"],
            transform=transformations,
            should_invert=should_invert
        )

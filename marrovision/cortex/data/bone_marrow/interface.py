import os
from pathlib import Path
import numpy as np
import torch
import torch.utils.data.dataloader
from .dataset import BoneMarrowDataset
from sklearn.model_selection import train_test_split

import marrovision.cortex.data.bone_marrow.transformations
from .sampler import BoneMarrowBalancedSampler
from .utilities import get_image_paths_per_label


def bone_marrow_cell_classification(
        data_dir: str,
        batch_size: int,
        test_ratio: float,
        balanced_sample_count_per_category: int,
        train_transformation: str,
        num_workers: int=None
):
    labels = [e for e in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, e)) and not e.startswith('.')]
    image_filepaths_per_label = {
        x: get_image_paths_per_label(data_dir, x) for x in labels
    }

    filepaths_per_label = dict(train=dict(), test=dict())
    for label in image_filepaths_per_label:
        filepaths_per_label['train'][label], filepaths_per_label['test'][label] = train_test_split(image_filepaths_per_label[label], test_size=test_ratio)

    datasets = {x: BoneMarrowDataset(
        filepaths_per_label[x],
        transform=getattr(marrovision.cortex.data.bone_marrow.transformations, train_transformation)() if x == 'train' else marrovision.cortex.data.bone_marrow.transformations.eval_transform_1()
    ) for x in ['train', 'test']}

    if balanced_sample_count_per_category is None:
        sampler = None
    else:
        sampler = BoneMarrowBalancedSampler(
            dataset=datasets['train'],
            number_of_samples_per_class=balanced_sample_count_per_category)

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        sampler=sampler if x == 'train' else None,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=((x == 'train') and (sampler is None))) for x in ['train', 'test']}

    assert dataloaders['test'].dataset.label_layout == dataloaders['train'].dataset.label_layout

    return dataloaders

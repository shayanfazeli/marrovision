import os
import torch
import torch.utils.data.dataloader
from .dataset import BoneMarrowDataset
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import marrovision.cortex.data.bone_marrow.transformations
from .sampler import BoneMarrowBalancedSampler


def get_image_paths_per_label(dataset_root, label):
    filepaths = []
    for path in Path(os.path.join(dataset_root, label)).rglob('*.jpg'):
        if path.is_file():
            filepaths.append(path)
    return filepaths
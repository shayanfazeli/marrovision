import os
from pathlib import Path
import numpy as np
import gzip
import pickle
import torch
import torchvision
import torch.utils.data.dataloader

from marrovision.contrib.torchvision.transforms import RandomMixup, RandomCutmix
from .dataset import BoneMarrowDataset
from sklearn.model_selection import train_test_split

import marrovision.cortex.data.bone_marrow.transformations
from .sampler import BoneMarrowBalancedSampler, BoneMarrowBalancedDistributedSampler
from .utilities import get_image_paths_per_label


def bone_marrow_cell_classification(
        data_dir: str,
        batch_size: int,
        test_ratio: float,
        balanced_sample_count_per_category: int,
        train_transformation: str,
        num_workers: int=None,
        distributed: bool = False,
        start_epoch: int = 0,
        seed: int = 0,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        save_split_to_filepath: str = None
):
    labels = [e for e in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, e)) and not e.startswith('.')]
    image_filepaths_per_label = {
        x: get_image_paths_per_label(data_dir, x) for x in labels
    }

    if mixup_alpha == 0.0 and cutmix_alpha == 0.0:
        collate_fn = torch.utils.data.dataloader.default_collate
    else:
        mixup_transforms = []
        if mixup_alpha > 0.0:
            mixup_transforms.append(RandomMixup(21, p=1.0, alpha=mixup_alpha))
        if cutmix_alpha > 0.0:
            mixup_transforms.append(RandomCutmix(21, p=1.0, alpha=cutmix_alpha))
        if mixup_transforms:
            mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
            def collate_fn(batch):
                batch = torch.utils.data.dataloader.default_collate(batch)
                collated_batch = dict()
                collated_batch['image'], collated_batch['gt_score'] = mixupcutmix(batch['image'], batch['label_index'])
                return collated_batch
        else:
            collate_fn = torch.utils.data.dataloader.default_collate

    filepaths_per_label = dict(train=dict(), test=dict())
    for label in image_filepaths_per_label:
        filepaths_per_label['train'][label], filepaths_per_label['test'][label] = train_test_split(image_filepaths_per_label[label], test_size=test_ratio)

    if save_split_to_filepath is not None:
        with gzip.open(save_split_to_filepath, 'wb') as f:
            pickle.dump(filepaths_per_label, f)

    datasets = {x: BoneMarrowDataset(
        filepaths_per_label[x],
        transform=getattr(marrovision.cortex.data.bone_marrow.transformations, train_transformation)() if x == 'train' else marrovision.cortex.data.bone_marrow.transformations.eval_transform_1()
    ) for x in ['train', 'test']}

    sampler_per_mode = dict()
    if not distributed:
        sampler_per_mode['test'] = None
        if balanced_sample_count_per_category is None:
            sampler_per_mode['train'] = None
        else:
            sampler_per_mode['train'] = BoneMarrowBalancedSampler(
                dataset=datasets['train'],
                number_of_samples_per_class=balanced_sample_count_per_category)
    else:
        sampler_per_mode['test'] = None
        if balanced_sample_count_per_category is None:
            sampler_per_mode['train'] = torch.utils.data.distributed.DistributedSampler(
                datasets['train'],
                shuffle=True,
                seed=seed
            )
            sampler_per_mode['train'].set_epoch(start_epoch)
        else:
            sampler_per_mode['train'] = BoneMarrowBalancedDistributedSampler(
                dataset=datasets['train'],
                number_of_samples_per_class=balanced_sample_count_per_category)
            sampler_per_mode['train'].set_epoch(start_epoch)

    dataloaders = {x: torch.utils.data.DataLoader(
        datasets[x],
        sampler=sampler_per_mode[x],
        batch_size=batch_size,
        collate_fn=collate_fn if x == 'train' else torch.utils.data.dataloader.default_collate,
        num_workers=num_workers) for x in ['train', 'test']}

    assert dataloaders['test'].dataset.label_layout == dataloaders['train'].dataset.label_layout

    return dataloaders

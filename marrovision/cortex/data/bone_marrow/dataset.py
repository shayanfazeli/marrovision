from typing import List, Dict
import os
import mmcv
import numpy
import gzip
import pickle
import torch.utils.data.dataset
import torchvision.transforms as transforms
from pathlib import Path
from typing import Tuple, Dict, Any, List
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class BoneMarrowDataset(torch.utils.data.dataset.Dataset):
    """
    """
    def __init__(
            self,
            image_filepaths_per_label,
            transform
    ):
        self.image_filepaths_per_label = image_filepaths_per_label
        self.label_layout = sorted(list(image_filepaths_per_label.keys()))
        self.image_count_per_label = {x: len(self.image_filepaths_per_label[x]) for x in self.image_filepaths_per_label}
        self.transform = transform
        self.filepaths = []
        self.labels = []
        for label in self.label_layout:
            self.filepaths.extend(self.image_filepaths_per_label[label])
            self.labels.extend([label] * self.image_count_per_label[label])

        self.label_indices = [self.label_layout.index(x) for x in self.labels]

    def __getitem__(self, index):
        x = mmcv.imread(str(self.filepaths[index]), channel_order='rgb')#
        x = self.transform(x)

        return {'image': x, 'label': self.labels[index], 'label_index': self.label_indices[index]}

    def __len__(self):
        return len(self.filepaths)
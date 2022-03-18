import os
import pdb
import sys
from datetime import datetime
from typing import Dict, Any
from overrides import overrides
import abc
from tqdm import tqdm
import numpy
import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.utils.data.dataloader
import apex
from apex.parallel.LARC import LARC

import logging

from marrovision.cortex.evaluation.metrics import compute_all_classification_metrics
from marrovision.cortex.trainer.base import TrainerBase
from .standard import StandardClassificationTrainer

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StandardSemiSupervisedClassificationTrainer(StandardClassificationTrainer, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super(StandardSemiSupervisedClassificationTrainer, self).__init__(*args, **kwargs)

    def buffering_initialization(self):
        """
        Initializing the buffering mechanism.
        """
        self.buffer = {
            'history': {
                'train': [],
                'test': []
            },
        }
        self.epoch_y = []
        self.epoch_y_hat = []
        self.epoch_y_score = []
        self.epoch_losses = []
        self.epoch_gt_score = []

    def iteration_buffering(self, mode: str, info_bundle: Dict[str, Any]):
        """
        Buffering the data for the current iteration.
        """
        if mode == 'train':
            self.epoch_gt_score.append(info_bundle['model_outputs']['gt_score'].data.cpu().numpy())
            self.epoch_y_score.append(info_bundle['model_outputs']['y_score'].data.cpu().numpy())
            self.epoch_losses.append({k: info_bundle['loss_outputs'][k].item() for k in info_bundle['loss_outputs'].keys()})
        else:
            self.epoch_y.append(info_bundle['model_outputs']['label_index'].data.cpu().numpy())
            self.epoch_y_hat.append(info_bundle['model_outputs']['y_hat'].data.cpu().numpy())
            self.epoch_y_score.append(info_bundle['model_outputs']['y_score'].data.cpu().numpy())
            self.epoch_losses.append({k: info_bundle['loss_outputs'][k].item() for k in info_bundle['loss_outputs'].keys()})

    def buffering_reset(self):
        """
        Resetting the buffering mechanism.
        """
        self.epoch_y = []
        self.epoch_y_hat = []
        self.epoch_y_score = []
        self.epoch_losses = []
        self.epoch_gt_score = []

    def process_buffers(self, mode: str, epoch_index: int):
        stats = dict()
        if mode == 'test':
            labels = self.dataloaders['test'].dataset.label_layout
            label_indices = list(range(len(labels)))
            stats = {'label_layout': {
                'labels': labels,
                'label_indices': label_indices
            }}

            # - getting the label layout
            stats.update(compute_all_classification_metrics(
                epoch_y=numpy.concatenate(self.epoch_y, axis=0),
                epoch_y_hat=numpy.concatenate(self.epoch_y_hat, axis=0),
                epoch_y_score=numpy.concatenate(self.epoch_y_score, axis=0),
                labels=label_indices))

        # - adding the loss stats
        for k in self.epoch_losses[0].keys():
            loss_values_throughout_epoch = [e[k] for e in self.epoch_losses]
            stats[f'loss_stats_for_{k}'] = {
                'mean': numpy.mean(loss_values_throughout_epoch),
                'median': numpy.median(loss_values_throughout_epoch),
                'std': numpy.std(loss_values_throughout_epoch),
                'min': numpy.min(loss_values_throughout_epoch),
                'max': numpy.max(loss_values_throughout_epoch),
            }

        if (self.distributed_rank is None) or (self.distributed_rank == 0):
            logger.info(f"""
            Performance ~> Epoch {epoch_index} - [{mode}]
            ===========================
            
            {stats}
            
            ===========================
            """)

        stats.update({'mode': mode, 'epoch_index': epoch_index})
        self.buffer['history'][mode].append(stats)
        self.buffering_reset()
        return stats

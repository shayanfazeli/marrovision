import os
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

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class StandardClassificationTrainer(TrainerBase, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        super(StandardClassificationTrainer, self).__init__(*args, **kwargs)
        self.distributed_initializations()

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

    def iteration_buffering(self, mode: str, info_bundle: Dict[str, Any]):
        """
        Buffering the data for the current iteration.
        """
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

    def distributed_initializations(self):
        if not self.config['args']['dist_url'] == 'none':

            self.optimizer = LARC(optimizer=self.optimizer, trust_coefficient=0.001, clip=False)
            logger.info("\t~> LARC optimizer initialization done.")
            # torch.distributed.init_process_group(
            #     backend="nccl",
            #     init_method=self.config['args']['dist_url'],
            #     world_size=self.config['args']['world_size'],
            #     rank=self.config['args']['rank']
            # )
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            logger.info("\t~> model batch norm synchronization is done.")

    def train_epoch(self, epoch_index: int) -> Dict[str, Any]:
        """
        training mechanism for the entire epoch.
        """
        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.dataloaders['train'])):
            self.optimizer.zero_grad()
            out = self.model(batch, mode='train')
            self.iteration_buffering(mode='train', info_bundle=out)

            # - computing the total loss by summing all the loss components
            loss = 0
            for loss_name, loss_value in out['loss_outputs'].items():
                loss = loss + loss_value

            if torch.isnan(loss):
                logger.error("NaN loss encountered. Exiting.")
                sys.exit(1)

            # - backpropagation
            loss.backward()

            self.optimizer.step()
            if self.iter_scheduler is not None:
                self.iter_scheduler.step()
        if self.epoch_scheduler is not None:
            self.epoch_scheduler.step()
        self.process_buffers(mode='train', epoch_index=epoch_index)

    def process_buffers(self, mode: str, epoch_index: int):
        labels = self.dataloaders['train'].dataset.label_layout
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

    def evaluate_epoch(self, epoch_index: int, ignore_checkpointing: bool = False) -> Dict[str, Any]:
        """
        epoch evaluation
        """
        self.model.eval()
        for batch_idx, batch in tqdm(enumerate(self.dataloaders['test'])):
            out = self.model(batch, mode='test')
            self.iteration_buffering(mode='test', info_bundle=out)
        stats = self.process_buffers(mode='test', epoch_index=epoch_index)
        if not ignore_checkpointing:
            self.checkpointing_dump(epoch_index=epoch_index)
            torch.save({'stats': stats}, os.path.join(self.config['checkpointing']['repo'], f'evaluation_{epoch_index}_{int(datetime.now().timestamp())}.pt'))
            logger.info(f"~> Evaluation is done, test stats are stored in {os.path.join(self.config['checkpointing']['repo'], f'evaluation_{epoch_index}_{int(datetime.now().timestamp())}.pt')}")

    def building_blocks(self) -> None:
        """
        Building the blocks of the trainer model.
        """
        # - optimizer
        self.optimizer = getattr(torch.optim, self.config['optimizer']['type'])(
            self.model.parameters(), **self.config['optimizer']['args'])
        self.epoch_scheduler = getattr(torch.optim.lr_scheduler, self.config['epoch_scheduler']['type'])(
            optimizer=self.optimizer, **self.config['epoch_scheduler']['args']
        ) if 'epoch_scheduler' in self.config else None
        self.iter_scheduler = getattr(torch.optim.lr_scheduler, self.config['iter_scheduler']['type'])(
            optimizer=self.optimizer, **self.config['iter_scheduler']['args']) if 'iter_scheduler' in self.config else None

    def checkpointing_dump(self, epoch_index: int):
        """
        checkpointing dump
        """

        if not self.config['args']['dist_url'] == 'none':
            if self.config['args']['rank'] != 0:
                return
        repo = os.path.abspath(self.config['checkpointing']['repo'])
        os.makedirs(repo, exist_ok=True)
        if 'checkpointing_epoch_interval' in self.config['checkpointing']:
            interval = self.config['checkpointing']['checkpointing_epoch_interval']
        else:
            interval = None

        iter_scheduler_state = None if self.iter_scheduler is None else self.iter_scheduler.state_dict()
        epoch_scheduler_state = None if self.epoch_scheduler is None else self.epoch_scheduler.state_dict()
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_scheduler': iter_scheduler_state,
            'epoch_scheduler': epoch_scheduler_state,
        }

        if interval is not None and epoch_index % interval == 0:
            torch.save(self.buffer['history'], os.path.join(repo, f'stats_epoch-{epoch_index}.pth'))
            torch.save(ckpt, os.path.join(repo, f'ckpt_epoch-{epoch_index}.pth'))
        torch.save(self.buffer['history'], os.path.join(repo, f'stats_latest.pth'))
        torch.save(ckpt, os.path.join(repo, f'ckpt_latest.pth'))

    def checkpointing_resume(self):
        repo = os.path.abspath(self.config['checkpointing']['repo'])
        if os.path.isfile(os.path.join(repo, 'ckpt_latest.pt')):
            assert os.path.isfile(os.path.join(repo, 'stats_latest.pt')), 'stats_latest.pt not found'
            ckpt = torch.load(os.path.join(repo, 'ckpt_latest.pt'), map_location='cpu')
            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt['iter_scheduler'] is not None:
                assert self.iter_scheduler is not None, 'iter_scheduler not found'
                self.iter_scheduler.load_state_dict(ckpt['iter_scheduler'])
            if ckpt['epoch_scheduler'] is not None:
                assert self.epoch_scheduler is not None, 'epoch_scheduler not found'
                self.epoch_scheduler.load_state_dict(ckpt['epoch_scheduler'])

            self.buffer['history'] = torch.load(os.path.join(repo, 'stats_latest.pt'), map_location='cpu')
            self.start_epoch = self.buffer['history']['train'][-1]['epoch_index'] + 1

    @overrides
    def eval(self):
        self.checkpointing_resume()
        self.evaluate_epoch(epoch_index=self.start_epoch-1, ignore_checkpointing=True)
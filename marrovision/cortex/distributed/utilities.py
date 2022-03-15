from typing import Union, Tuple
import argparse

import apex
import apex.parallel
import torch
import torch.nn
import torch.optim
import torch.distributed
import torch.utils.data.dataloader

import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def distributed_model(
        model: torch.nn.Module,
        args: argparse.Namespace,
        optimizer: torch.optim.Optimizer = None,
        # mixed_precision: bool = False
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, torch.optim.Optimizer]]:
    """
    Parameters
    ----------
    model: `torch.nn.Module`, required
        The main model

    args: `argparse.Namespace`, required
        The parsed arguments

    optimizer: `torch.optim.Optimizer`, optional (default=None)
        If the mixed precision mode is on, and only then, the optimizer must be passed so that the apex initialization
        be handled properly by this method.

    Returns
    ----------
    If the `mixed_precision` is `True`, it will return the prepared model and optimizer for further processing. If not,
    the now `torch.nn.parallel.DistributedDataParallel` wrapped model will be returned.
    """
    if not mixed_precision:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        assert optimizer is None, f"the caller does NOT need to pass the optimizer as the mixed precision training mode is {mixed_precision}"
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # process_group = apex.parallel.create_syncbn_process_group(0)
        # model = apex.parallel.convert_syncbn_model(model, process_group=process_group)

        # - preparation of apex
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1")
        logger.info("Initializing mixed precision done for the distributed setting.")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)

    if not mixed_precision:
        return model
    else:
        return model, optimizer

#!/usr/bin/env python
import os
import torch
import torch.distributed
import argparse
import logging
import marrovision.cortex.data as data_lib
import marrovision.cortex.model as model_lib
import marrovision.cortex.trainer as trainer_lib
from marrovision.utilities.checkpointing.get_start_epoch import get_start_epoch
from marrovision.utilities.device import get_device
from marrovision.utilities.io.files_and_folders import clean_folder
from marrovision.utilities.randomization.seed import fix_random_seeds
from marrovision.contrib.mmcv import Config
from marrovision.utilities.argument_parsing.train import train_parser


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main(args: argparse.Namespace) -> None:
    """
    Parameters
    ----------
    args: `argparse.Namespace`, required
        The arguments parsed from the command line.
    """
    # - fixing random seed
    fix_random_seeds(seed=args.seed)

    # - getting the configuration
    config = dict(Config.fromfile(args.config))
    config['trainer']['config'].update({'args': vars(args)})
    if args.clean:
        clean_folder(folder_path=config['trainer']['config']['checkpointing']['repo'])
    if args.fold_index:
        assert config['data']['kfold_config'] is not None
        config['data']['kfold_config']['shuffle'] = True
        config['data']['kfold_config']['random_state'] = args.seed
        config['data']['fold_index'] = args.fold_index
        config['trainer']['config']['checkpointing']['repo'] = os.path.join(config['trainer']['config']['checkpointing']['repo'] , f'fold_{args.fold_index}')

    # - distributed
    if not args.dist_url == 'none':
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.device = args.rank % torch.cuda.device_count()
        logger.info(f" ~> distributed mode is activated with the rank of {args.rank} and world_size of {args.world_size}")

        torch.distributed.init_process_group(
            backend="nccl",
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        config['data']['args']['distributed'] = True
    else:
        config['data']['args']['distributed'] = False
    config['data']['args']['save_split_to_filepath'] = os.path.join(config['trainer']['config']['checkpointing']['repo'], 'train_test_split.pkl.gz')

    # - getting device
    device = get_device(device=args.device)

    # - getting the dataloaders
    logger.info("~> preparing the dataloaders...\n")

    config['data']['args']['start_epoch'] = get_start_epoch(config)
    config['data']['args']['seed'] = args.seed

    # - dumping config and args
    checkpointing_repo = config['trainer']['config']['checkpointing']['repo']
    os.makedirs(checkpointing_repo, exist_ok=True)
    torch.save(config, os.path.join(checkpointing_repo, f'config_and_args.pt'))

    dataloaders = getattr(
        data_lib,
        config['data']['interface']
    )(**config['data']['args'])

    # - preparing the model
    logger.info(f"~> preparing the model (model type: {config['model']['type']})...\n")
    model = getattr(model_lib, config['model']['type'])(
        config=config['model']['config'],
        device=device
    ).to(device)

    # - trainer
    logger.info(f"~> preparing the trainer (trainer type: {config['trainer']['type']})...\n")
    trainer = getattr(trainer_lib, config['trainer']['type'])(
        config=config['trainer']['config'],
        dataloaders=dataloaders,
        model=model,
        device=device
    )

    # - starting the training procedure
    if not args.eval:
        logger.info(f"~> starting training sequence...\n")
        trainer.train()
    else:
        logger.info(f"~> starting evaluation...\n")
        trainer.eval()


if __name__ == "__main__":
    # - getting the arguments needed for training
    parser = train_parser()
    # - parsing the arguments
    args = parser.parse_args()

    # - running the training tool
    main(args=args)

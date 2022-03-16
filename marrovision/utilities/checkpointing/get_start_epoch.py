import os
import torch
import logging

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_start_epoch(config):
    config = config['trainer']['config']

    repo = os.path.abspath(config['checkpointing']['repo'])
    if os.path.isfile(os.path.join(repo, 'ckpt_latest.pth')):
        assert os.path.isfile(os.path.join(repo, 'stats_latest.pth')), 'stats_latest.pt not found'
        buffer_history = torch.load(os.path.join(repo, 'stats_latest.pth'), map_location='cpu')
        start_epoch = buffer_history['test'][-1]['epoch_index'] + 1
        logger.info("\t~> [sampler epoch set will take place on epoch {}]".format(start_epoch))
    else:
        start_epoch = 0
    return start_epoch
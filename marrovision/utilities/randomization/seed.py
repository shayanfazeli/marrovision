import random
import torch
import torch.backends.cudnn
import numpy


def fix_random_seeds(seed: int) -> None:
    """
    Parameters
    ----------
    seed: `int`, required
        The integer seed
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)

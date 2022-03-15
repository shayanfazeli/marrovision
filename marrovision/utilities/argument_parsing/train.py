import argparse


def train_parser() -> argparse.ArgumentParser:
    """
    Returns the argument parser for the train command.
    """
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "config", help="Path to the config file.", type=str)
    parser.add_argument("--device", default=-1, help="Device to use.", type=int)
    parser.add_argument("--seed", default=42, help="Random seed.", type=int)
    parser.add_argument("--clean", action="store_true", help="Restart training and erase previous checkpoint contents.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model (the latest checkpoint must exist in "
                                                            "the folder, and it will be used.")
    # -- automatically set arguments
    parser.add_argument("--dist_url", default="none", type=str,
                        help="""url used to set up distributed training; see 
                        https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--world_size", default=-1, type=int,
                        help="number of processes: it is set automatically and should not be passed as argument")
    parser.add_argument("--rank", default=0, type=int, help="""rank of this process:
                                    it is set automatically and should not be passed as argument""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="this argument is not used and should be ignored")
    return parser

from typing import Iterable
import numpy
import torch.utils.data.sampler
import random
import math
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class BoneMarrowBalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, number_of_samples_per_class: int, seed: int = 0):
        assert number_of_samples_per_class > 0, "number_of_samples_per_class must be greater than 0"
        self.number_of_samples_per_class = number_of_samples_per_class
        self.filepaths = dataset.filepaths
        self.label_indices = numpy.array(dataset.label_indices)
        self.label_layout = dataset.label_layout
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterable[int]:
        indices = []
        for lbl_index in range(len(self.label_layout)):
            if lbl_index not in self.label_indices:
                raise ValueError("Label index {} is not in dataset".format(lbl_index))
            possible_indices = numpy.nonzero(self.label_indices == lbl_index)[0]
            indices.append(numpy.random.choice(possible_indices, size=self.number_of_samples_per_class, replace=True))
        indices = numpy.concatenate(indices, axis=0).astype('int').tolist()

        random.seed(self.epoch + self.seed)
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.number_of_samples_per_class * len(self.label_layout)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class BoneMarrowBalancedDistributedSampler(torch.utils.data.sampler.Sampler[T_co]):
    def __init__(self, dataset: torch.utils.data.Dataset, number_of_samples_per_class: int, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        assert number_of_samples_per_class > 0, "number_of_samples_per_class must be greater than 0"
        self.number_of_samples_per_class = number_of_samples_per_class
        self.filepaths = dataset.filepaths
        self.label_indices = numpy.array(dataset.label_indices)
        self.label_layout = dataset.label_layout
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        total_length = self.number_of_samples_per_class * len(self.label_layout)
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and total_length % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (total_length - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(total_length / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        indices = []
        for lbl_index in range(len(self.label_layout)):
            if lbl_index not in self.label_indices:
                raise ValueError("Label index {} is not in dataset".format(lbl_index))
            possible_indices = numpy.nonzero(self.label_indices == lbl_index)[0]
            indices.append(numpy.random.choice(possible_indices, size=self.number_of_samples_per_class, replace=True))
        indices = numpy.concatenate(indices, axis=0).astype('int').tolist()
        random.seed(self.epoch + self.seed)
        random.shuffle(indices)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        print("Rank {}: {}".format(self.rank, len(indices)))
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
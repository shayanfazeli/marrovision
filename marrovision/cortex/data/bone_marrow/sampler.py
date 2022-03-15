from typing import Iterable
import numpy
import torch.utils.data.sampler
import random


class BoneMarrowBalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, number_of_samples_per_class: int):
        assert number_of_samples_per_class > 0, "number_of_samples_per_class must be greater than 0"
        self.number_of_samples_per_class = number_of_samples_per_class
        self.filepaths = dataset.filepaths
        self.label_indices = numpy.array(dataset.label_indices)
        self.label_layout = dataset.label_layout

    def __iter__(self) -> Iterable[int]:
        indices = []
        for lbl_index in range(len(self.label_layout)):
            if lbl_index not in self.label_indices:
                raise ValueError("Label index {} is not in dataset".format(lbl_index))
            possible_indices = numpy.nonzero(self.label_indices == lbl_index)[0]
            indices.append(numpy.random.choice(possible_indices, size=self.number_of_samples_per_class, replace=True))
        indices = numpy.concatenate(indices, axis=0).astype('int').tolist()
        random.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.number_of_samples_per_class * len(self.label_layout)

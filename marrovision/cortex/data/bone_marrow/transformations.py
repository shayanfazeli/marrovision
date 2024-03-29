import torch
import torchvision.transforms as transforms


def train_transform_1():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(250, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=[0.56303613, 0.49592108, 0.73533199], std=[0.24209296, 0.28346966, 0.17672639])
    ])


def train_transform_2():
    # -  same as 1, except with imagenet normalization
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomResizedCrop(250, scale=(0.7, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])


def eval_transform_1():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56303613, 0.49592108, 0.73533199], std=[0.24209296, 0.28346966, 0.17672639])
    ])
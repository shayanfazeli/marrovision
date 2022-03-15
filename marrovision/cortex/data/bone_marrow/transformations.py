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


def eval_transform_1():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.56303613, 0.49592108, 0.73533199], std=[0.24209296, 0.28346966, 0.17672639])
    ])
import numpy as np
import torch
import math
import copy

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2471, 0.2435, 0.2616]
cifar100_mean = [0.5071, 0.4867, 0.4408]
cifar100_std = [0.2675, 0.2565, 0.2761]


def weak_augmentation(mean, std, train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.train_batch:
        num_expand_x = math.ceil(
            args.train_batch * args.iter_per_epoch / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


def get_cifar10(args, root):
    transform_labeled = weak_augmentation(cifar10_mean, cifar10_std, True)
    transform_val = weak_augmentation(cifar10_mean, cifar10_std, False)
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=transform_labeled, is_strong_augment=True)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):

    transform_labeled = weak_augmentation(cifar100_mean, cifar100_std, True)

    transform_val = weak_augmentation(cifar100_mean, cifar100_std, False)

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=transform_labeled, is_strong_augment=True)

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, is_strong_augment=False, strong_augment=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.transform = transform
            self.is_strong_augment = is_strong_augment
            if self.is_strong_augment:
                if strong_augment is None:
                    self.strong_augment = transforms.RandAugment(3, 4)
                else:
                    self.strong_augment = strong_augment

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_strong = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)

        if self.is_strong_augment:
            img_strong = self.strong_augment(img.byte())
            return torch.tensor(img), torch.tensor(img_strong), target.long()

        return torch.tensor(img), target.long()


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, is_strong_augment=False, strong_augment=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.transform = transform
            self.is_strong_augment = is_strong_augment
            if self.is_strong_augment:
                if strong_augment is None:
                    self.strong_augment = transforms.RandAugment(1, 2)
                else:
                    self.strong_augment = strong_augment

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img_strong = img

        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)

        if self.is_strong_augment:
            img_strong = self.strong_augment(img)
            print('Weak = {}, Strong = {}'.format(torch.tensor(img).dtype, torch.tensor(img_strong).dtype))
            return torch.tensor(img), torch.tensor(img_strong), target.long()

        return torch.tensor(img), target.long()

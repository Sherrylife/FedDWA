import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from torch.utils.data import Subset
from torchvision.datasets import DatasetFolder, ImageFolder

class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = imagefolder_obj.samples[self.dataidxs]
        else:
            self.samples = imagefolder_obj.samples

        for image in self.samples:
            if self.transform is not None:
                self.data.append(self.transform(self.loader(image[0])).numpy())
            else:
                self.data.append(self.loader(image[0]))
            self.targets.append(int(image[1]))
        self.indices = range(len(self.targets))

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)



"""Just for different privacy,especially for DPSGD method"""
class RandomSampledDataset2(Dataset):
    def __init__(self, dataset, q=1.0):
        self.dataset = dataset
        self.indexes = range(len(self.dataset))
        self.length = int(len(self.indexes) * q)
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image, label = self.dataset[self.random_indexes[item]]
        self.count += 1
        if self.count == self.length:
            self.reset()
        return image, label

    def reset(self):
        self.count = 0
        self.random_indexes = np.random.choice(self.indexes, self.length, replace=False)



class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''

    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform


    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.subset_transform:
            x = self.subset_transform(x)

        return x, y


class CustomDataset(Dataset):
    def __init__(self, dataset, subset_transform=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        return image, label


class PartitionedDataset(Dataset):
    def __init__(self, dataset, indexes, subset_transform=None):
        self.dataset = dataset
        self.indexes = list(indexes)
        self.subset_transform = subset_transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, item):
        image, label = self.dataset[self.indexes[item]]

        if self.subset_transform:
            image = self.subset_transform(image)

        return image, label

def load_dataset(name: str, sample_rate=1.0):

    if name == 'cifar100tpds':
        return load_cifar100_liketpds(sample_rate)
    elif name == 'cifar10tpds':
        return load_cifar_liketpds(sample_rate)
    elif name == 'cinic-10':
        return load_cinic_10()
    elif name == 'tiny_ImageNet':
        return load_tiny_imagenet()
    else:
        raise NotImplementedError


def load_cinic_10(data_dir='./data/cinic-10/', sample_rate=1.0):
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    transform_cinic_10_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std),
    ])
    transform_cinic_10_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std),
    ])
    dl_obj = ImageFolder_custom
    dataset_train = dl_obj(data_dir + 'train/', transform=transform_cinic_10_train)
    dataset_test = dl_obj(data_dir + 'test/', transform=transform_cinic_10_test)
    return dataset_train, dataset_test




def load_tiny_imagenet(data_dir='./data/tiny-imagenet-200/', sample_rate=1.0):
    transform_tiny_imagenet_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_tiny_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # dataset_train = datasets.ImageNet(data_dir, train=True, download=True, transform=transform_tiny_imagenet_train)
    # dataset_test = datasets.ImageNet(data_dir, train=False, download=True, transform=transform_tiny_imagenet_test)
    # print('dataset_train[0][1]=',dataset_train.__getitem__(0)[0].shape)
    dl_obj = ImageFolder_custom
    dataset_train = dl_obj(data_dir + 'train/', transform=transform_tiny_imagenet_train)
    dataset_test = dl_obj(data_dir + 'val/', transform=transform_tiny_imagenet_test)
    return dataset_train, dataset_test





def load_cifar100_liketpds(sample_rate=1.0):
    train_data = datasets.CIFAR100('./data/cifar100', train=True, download=True, transform=transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_data = datasets.CIFAR100('./data/cifar100', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    # train_idcs = np.random.RandomState(seed=42).permutation(len(train_data))
    # train_idcs = train_idcs[:int(sample_rate * len(train_data))]
    # test_idcs = np.random.RandomState(seed=42).permutation(len(test_data))
    # test_idcs = test_idcs[:int(sample_rate * len(test_data))]
    # train_set = CustomSubset(train_data, train_idcs)
    # test_set = CustomSubset(test_data, test_idcs)

    return train_data, test_data

def load_cifar_liketpds(sample_rate=1.0):
    train_data = datasets.CIFAR10('./data/cifar10', train=True, download=True, transform=transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    test_data = datasets.CIFAR10('./data/cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]))
    # train_idcs = np.random.RandomState(seed=42).permutation(len(train_data))
    # train_idcs = train_idcs[:int(sample_rate * len(train_data))]
    # test_idcs = np.random.RandomState(seed=42).permutation(len(test_data))
    # test_idcs = test_idcs[:int(sample_rate * len(test_data))]
    # train_set = CustomSubset(train_data, train_idcs)
    # test_set = CustomSubset(test_data, test_idcs)

    return train_data, test_data

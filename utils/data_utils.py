import numpy as np
import torch
import random
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
from utils.dataset import *

def noniid_type8(datasetname, dataset, num_users, num_classes=10, sample_assignment=None, test=False,logger = None):
    """
    Create non-iid client data for MNIST or CIFAR10 dataset, this function can
    only ensure that each user is assigned no more than two (default value, it can be modified
    through the variable "each_client_class_num") class tags, not that each user is assigned two class tags
    :param dataset:
    :param num_users: client number.
    :param num_classes: the number of categories of the dataset
    :return:
    """

    each_client_class_num = 2


    dataset_image = []
    dataset_label = []
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset.targets), shuffle=False)

    for _, data in enumerate(dataloader, 0):
        dataset_data, dataset_targets = data
    dataset_image.extend(np.array(dataset_data))
    dataset_label.extend(np.array(dataset_targets))
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    # each client obtain no more than 2 shards
    num_shards = int(num_users * each_client_class_num)
    order = np.argsort(dataset_label)
    x_sorted = dataset_image[order]
    y_sorted = dataset_label[order]

    n_shards = num_users * 2
    # split data into shards of (mostly) the same index
    x_shards = np.array_split(x_sorted, n_shards)
    y_shards = np.array_split(y_sorted, n_shards)
    if sample_assignment is None:
        sample_assignment = np.array_split(np.random.permutation(n_shards), num_users)

    x_sharded = []
    y_sharded = []
    for w in range(num_users):
        x_sharded.append(np.concatenate([x_shards[i] for i in sample_assignment[w]]))
        y_sharded.append(np.concatenate([y_shards[i] for i in sample_assignment[w]]))

    # data split
    data = []
    for i in range(num_users):
        X, y = x_sharded[i], y_sharded[i]
        logger.info(np.unique(y))
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)

        data.append([(x, y) for x, y in zip(X, y)])

    return data, sample_assignment


def noniid_type9(datasetname, trainset, testset, num_users, num_classes=10,dirichlet_alpha=0.1, least_samples=20,logger = None):
    """
    Use the Dirichlet distribution to divide the dataset. The implementation of code is mainly refer to the website:
    "https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py", and make some modifications.
    :param datasetname: the name of the datsaet
    :param trainset:
    :param testset:
    :param num_users: the number of clients
    :param alpha_dirichlet: the hyper-parameter to control the non-iid degree
    :param num_class: the number of classes for the dataset
    :param least_samples: the minimum samples that each client should have
    """

    dataset_image = []
    dataset_label = []
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.targets), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.targets), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset_data, trainset_targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset_data, testset_targets = test_data
    dataset_image.extend(np.array(trainset_data))
    dataset_image.extend(np.array(testset_data))
    dataset_label.extend(np.array(trainset_targets))
    dataset_label.extend(np.array(testset_targets))

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(len(dataset_label))
    # sort labels
    idxs_labels = np.vstack((idxs, dataset_label))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    min_size = 0

    K = num_classes
    N = len(dataset_label)

    while min_size < least_samples:
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(dataset_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_users):
        dict_users[j] = idx_batch[j]

    # train test split
    train_data, test_data = [], []
    for i in range(num_users):
        logger.info(f'label types of the {i} client are:{np.unique(dataset_label[list(dict_users[i])])}')
        X_train, X_test, y_train, y_test = train_test_split(dataset_image[list(dict_users[i])],
                                                            dataset_label[list(dict_users[i])], train_size=0.8,
                                                            shuffle=True)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        train_data.append([(x, y) for x, y in zip(X_train, y_train)])
        test_data.append(([(x, y) for x, y in zip(X_test, y_test)]))

    return train_data, test_data


def noniid_type10(datasetname, dataset, num_users, num_types, ratio, num_classes=10,logger = None):
    """
    This implementation refers to the description of
    "Personalized Cross-Silo Federated Learninn on Non-IID Data" (FedAMP)
    :param datasetname:
    :param dataset:
    :param num_users:
    :param num_classes: the number of categories for the dataset,e.g., for mnist, num_classes=10
    :param num_types: the number of major categories per client
    :param ratio: The proportion of the number of major categories
            to the total number of categories
    """

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset.targets), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        dataset_image, dataset_label = train_data
    # dataset_image = np.array(trainset_data)
    # dataset_label = np.array(trainset_targets)
    order = torch.randperm(dataset_image.shape[0])
    image_random = dataset_image[order]
    label_random = dataset_label[order]
    offset = int(dataset_image.shape[0] * ratio)
    image_class = image_random[:offset]
    label_class = label_random[:offset]
    image_s = image_random[offset:]
    label_s = label_random[offset:]

    order = torch.argsort(label_class)
    x_sorted = image_class[order]
    y_sorted = label_class[order]
    # split data into bum class
    x_shards = torch.tensor_split(x_sorted, num_types)
    y_shards = torch.tensor_split(y_sorted, num_types)

    x_num, y_num = [], []
    for i in range(num_types):
        order = torch.randperm(x_shards[i].shape[0])
        x_order = x_shards[i][order]
        y_order = y_shards[i][order]
        if i == num_types - 1:
            x_num += torch.tensor_split(x_order, (num_users - (num_types -1) * int(num_users/num_types)))
            y_num += torch.tensor_split(y_order, (num_users - (num_types -1) * int(num_users/num_types)))
        else:
            x_num += torch.tensor_split(x_order, (int(num_users / num_types)))
            y_num += torch.tensor_split(y_order, (int(num_users / num_types)))

    x_split_all = torch.tensor_split(image_s, num_users)
    y_split_all = torch.tensor_split(label_s, num_users)


    data = []
    for i in range(num_users):
        X, y = [], []
        X.extend(torch.cat((x_num[i], x_split_all[i])))
        y.extend(torch.cat((y_num[i], y_split_all[i])))
        logger.info(f'label types of the {i} client are:{torch.unique(torch.cat((y_num[i], y_split_all[i])))}')

        data.append([(x, y) for x, y in zip(X, y)])

    return data


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


def dirichlet_noniid(dataset, num_users=10, dirichlet_alpha = 100, sample_matrix_test=None, test=False):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter alpha
    The implementation of code is refer to the article:
    "A Bayesian Federated Learning Framework with Online Laplace Approximation"
    """
    train_idcs = np.random.permutation(len(dataset))
    train_labels = np.array(dataset.targets)
    class_num = train_labels.max()+1

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs = np.arange(len(dataset.targets))
    labels = np.asarray(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    class_lableidx = [idxs_labels[:, idxs_labels[1, :] == i][0, :] for i in range(class_num)]

    if test is True and sample_matrix_test is not None:
        sample_matrix = sample_matrix_test
    else:
        sample_matrix = np.random.dirichlet([dirichlet_alpha for _ in range(num_users)], class_num).T
    class_sampe_start = [0 for i in range(class_num)]

    for i in range(num_users):
        rand_set, class_sampe_start = sample_rand(sample_matrix[i], class_lableidx, class_sampe_start)
        dict_users[i] = rand_set

    return dict_users, sample_matrix

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()


def sample_rand(rand, class_lableidx, class_sampe_start):
    class_sampe_end = [start + int(len(class_lableidx[sidx]) * rand[sidx]) for sidx, start in
                       enumerate(class_sampe_start)]
    rand_set = np.array([], dtype=np.int32)
    for eidx, rand_end in enumerate(class_sampe_end):
        rand_start = class_sampe_start[eidx]
        if rand_end <= len(class_lableidx[eidx]):
            rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:rand_end]], axis=0)

        else:
            if rand_start < len(class_lableidx[eidx]):
                rand_set = np.concatenate([rand_set, class_lableidx[eidx][rand_start:]], axis=0)
            else:
                rand_set = np.concatenate(
                    [rand_set, random.sample(class_lableidx[eidx], rand_end - rand_start + 1)], axis=0)
    if rand_set.shape[0] == 0:
        rand_set = np.concatenate([rand_set, class_lableidx[0][0:1]], axis=0)
    return rand_set, class_sampe_end

def split_index(label_start_index, each_class_num, num_shards, num_classes=10):
    """
    :param label_start_index: the start index for each class
    :param each_class_num: the data number of each class
    :param num_shards: the total number of shard
    :param num_classes:
    :return label_start_index: the start index of each shard
    """

    each_class_shards_num = int(num_shards / num_classes) # how many shards each class should be divided
    for i in range(num_classes):
        num_samples = np.random.randint(low=1, high=max(int(each_class_num[i]/each_class_shards_num), 1), size=(each_class_shards_num-1)).tolist()
        num_samples.append(each_class_num[i]-sum(num_samples))
        sum_num_samples = np.array([np.sum(num_samples[:j+1])+np.sum(each_class_num[:i]) for j in range(len(num_samples)-1)])
        label_start_index = np.concatenate((label_start_index[:int(each_class_shards_num*i)+1], sum_num_samples, label_start_index[int(each_class_shards_num*i)+1:]))

    return label_start_index

def data_loader(datasetname, trainset, testset):
    """
    Convert the data from numpy to tensor, and mix the trainset and testset according the the sample rate.
    :param trainset:
    :param testset:
    """
    dataset_image = []
    dataset_label = []
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.indices), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.indices), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset_data, trainset_targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset_data, testset_targets = test_data
    dataset_image.extend(np.array(trainset_data))
    dataset_image.extend(np.array(testset_data))
    dataset_label.extend(np.array(trainset_targets))
    dataset_label.extend(np.array(testset_targets))
    sample_dataset_image = np.array(dataset_image)
    sample_dataset_label = np.array(dataset_label)
    return sample_dataset_image, sample_dataset_label


def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res
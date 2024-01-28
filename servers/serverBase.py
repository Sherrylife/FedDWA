import h5py
import torch
import os
import numpy as np
import copy
import time
import random
from torch import nn
import json
from pathlib import Path
from torchvision import transforms
from utils.data_utils import *
from utils.dataset import *
from torch.utils.data import DataLoader, TensorDataset

class ServerBase(object):
    def __init__(self, args, modelObj, run_times,logger):
        self.device = args.device
        self.dataset_name = args.dataset
        self.model_name = args.model
        self.algorithm_name = args.alg
        self.train_set = None
        self.test_set = None
        self.test_loader = None
        self.global_rounds = args.Tg
        self.current_global_round = 0
        self.local_steps = args.E
        self.batch_size = args.B
        self.learning_rate = args.lr
        self.evaluate_gap = 1 # how much round we will test the model
        self.times = args.times
        self.seed = args.seed

        self.sample_rate = args.sample_rate
        self.logger = logger
        self.feddwa_topk = args.feddwa_topk
        self.args = args

        self.num_types_noniid10 = args.num_types_noniid10
        self.ratio_noniid10 = args.ratio_noniid10

        self.num_clients = args.client_num
        self.client_join_ratio = args.client_frac
        self.join_clients_num = int(self.num_clients * self.client_join_ratio)
        self.algorithm_name = args.alg

        self.noniidtype = args.non_iidtype
        self.all_train_set = None
        self.all_test_set = None
        self.dirichlet_alpha = args.alpha_dir
        self.seed = args.seed
        self.num_classes = args.num_classes
        self.next_round = args.next_round


        self.global_model = copy.deepcopy(modelObj)

        self.clientsObj = []
        self.selected_clients_idx = []
        self.client_traindata_idx = []
        self.client_testdata_idx = []
        self.receive_client_models = []
        self.receive_client_datasize = []
        self.receive_client_weight = []

        self.client_test_acc_logs = []
        self.client_train_acc_logs = []
        self.client_test_loss_logs = [] # used for linear regression
        self.client_train_loss_logs = []
        self.client_mean_test_acc_logs = []


    def dataset_division(self):
        self.train_set, self.test_set = load_dataset(self.dataset_name, self.sample_rate)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        if self.noniidtype == 8:
            noniid_fn = noniid_type8
            self.all_train_set, assignment = noniid_fn(self.dataset_name, self.train_set, num_users=self.num_clients, logger=self.logger)
            self.all_test_set, _ = noniid_fn(self.dataset_name, self.test_set, num_users=self.num_clients,
                                             sample_assignment=assignment, test=True,logger=self.logger)
        elif self.noniidtype == 9:
            noniid_fn = noniid_type9
            self.all_train_set, self.all_test_set = noniid_fn(self.dataset_name, self.train_set, self.test_set,
                                                              num_users=self.num_clients, num_classes = self.num_classes,
                                                              dirichlet_alpha=self.dirichlet_alpha, logger=self.logger)
        elif self.noniidtype == 10:
            noniid_fn = noniid_type10
            self.all_train_set = noniid_fn(self.dataset_name, self.train_set, num_users=self.num_clients,num_types=self.num_types_noniid10, ratio=self.ratio_noniid10, logger=self.logger)
            self.all_test_set = noniid_fn(self.dataset_name, self.test_set, num_users=self.num_clients, num_types=self.num_types_noniid10, ratio=self.ratio_noniid10, logger=self.logger)
        else:
            raise NotImplementedError

    def set_clients(self, args, modelObj, clientObj):
        for idx in range(self.num_clients):
            client = clientObj(args,
                               id=idx,
                               modelObj=modelObj,
                               train_set=CustomDataset(self.all_train_set[idx]),
                               test_set=CustomDataset(self.all_test_set[idx]),
                               )
            self.clientsObj.append(client)

    def select_client(self, method=0):
        """
        Two methods to select client:
        1. random manner to select client
        2. robin manner to select client (usually used in different privacy)
        """
        if method ==0: # random select
            selected_clients_idx = list(np.random.choice(range(self.num_clients), int(self.client_join_ratio * self.num_clients), replace=False))
        else: # robin manner to select
            shard_size = self.num_clients * self.client_join_ratio
            shard_num = np.ceil(1 / self.client_join_ratio)
            shard_idx = self.current_global_round % shard_num

            start = shard_idx * shard_size
            end = min((shard_idx + 1) * shard_size, self.num_clients)
            end = max(end, start + 1)
            selected_clients_idx = range(int(start), int(end))
            self.current_global_round += 1

        return selected_clients_idx

    def test_global_data(self):
        """compute accuracy using the common one test set and global model"""
        correct = 0
        total = 0
        self.global_model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.global_model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        # save acc
        return acc

    def send_models(self):
        assert (len(self.selected_clients_idx) > 0)
        for idx in self.selected_clients_idx:
            self.clientsObj[idx].receive_models(self.global_model)

    def receive_models(self):
        assert(len(self.selected_clients_idx) > 0)
        # self.receive_client_models = [copy.deepcopy(self.clientsObj[idx].model) for idx in self.selected_clients_idx]
        self.receive_client_models = [self.clientsObj[idx].model for idx in self.selected_clients_idx]
        self.receive_client_datasize = np.array([self.clientsObj[idx].data_size for idx in self.selected_clients_idx])
        self.receive_client_weight = self.receive_client_datasize/self.receive_client_datasize.sum()

    def aggregated(self):
        "Base method is FedAvg"
        assert (len(self.selected_clients_idx) > 0)
        for param in self.global_model.parameters():
            param.data.zero_()
        for weight, client_model in zip(self.receive_client_weight, self.receive_client_models):
            for global_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
                global_param.data += client_param.data.clone() * weight

    def evaluate_acc(self, selected_all=False):
        """
        calclulate each client test acccuracy and then
        calclulate the weighted-mean accuracy
        """
        if selected_all == True:
            # test all clients
            acc_logs = []
            for idx in range(self.num_clients):
                client_test_acc = self.clientsObj[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in range(self.num_clients)])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc
        else:
            acc_logs = []
            for idx in self.selected_clients_idx:
                client_test_acc = self.clientsObj[idx].test_accuracy()
                acc_logs.append(client_test_acc)

            client_mean_test_acc = 0.0
            receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in self.selected_clients_idx])
            receive_client_weight = receive_client_datasize / receive_client_datasize.sum()
            for weight, acc in zip(receive_client_weight, acc_logs):
                client_mean_test_acc += weight * acc

        return acc_logs, client_mean_test_acc




    def save_results(self):
        result_path = "./results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.client_test_acc_logs) > 0):
            logs = Path('./logs_feddwa')
            filename = f'{self.dataset_name}_{self.algorithm_name}_model={self.model_name}_dwaToopK={self.feddwa_topk}_next={self.next_round}_C={self.client_join_ratio}_Tg={self.global_rounds}_N={self.num_clients}_lr={self.learning_rate}_E={self.local_steps}_noniid={self.noniidtype}_nType={self.num_types_noniid10}_ratio={self.ratio_noniid10}_alpha={self.dirichlet_alpha}_{self.seed}_{self.times}.json'
            store_data = {'test_acc':self.client_test_acc_logs, 'train_loss': self.client_train_loss_logs, 'test_weighted-mean_acc': self.client_mean_test_acc_logs}
            with (logs / filename).open('w', encoding='utf8') as f:
                json.dump(store_data, f)

    def print(self, test_acc, train_acc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Train Accurancy: {:.4f}".format(train_acc))
        print("Average Train Loss: {:.4f}".format(train_loss))



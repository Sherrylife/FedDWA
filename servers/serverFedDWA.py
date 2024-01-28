import copy
import time

import numpy as np
import torch

from clients.clientFedDWA import clientFedDWA
from servers.serverBase import ServerBase



class FedDWA(ServerBase):
    def __init__(self, args, modelObj, run_times,logger):
        super(FedDWA, self).__init__(args, modelObj, run_times,logger)
        self.dataset_division()
        self.set_clients(args, modelObj, clientFedDWA)
        # the initial send model for each client is the same
        self.send_client_models = None
        self.receive_client_next_models = []
        self.weight_matrix = None   # store the optimal weight
        self.feddwa_topk = args.feddwa_topk

    def send_models(self):
        """
        Override the "send_models" function of ServerBase,
        since in FedDWA, we need to send different models for each client
        """
        assert (len(self.selected_clients_idx) > 0)
        if self.send_client_models is not None:
            for idx, model_param in self.send_client_models.items():
                client_model = copy.deepcopy(self.send_client_models[idx])
                self.clientsObj[idx].receive_models(client_model)

    def receive_models(self):
        """
        Override the "receive_models" function of ServerBase,
        since in FedDWA, we need to receive two models for each client
        """
        assert (len(self.selected_clients_idx) > 0)
        self.receive_client_models = [{key: value for key, value in self.clientsObj[idx].model.named_parameters()} for idx in self.selected_clients_idx]
        self.receive_client_datasize = np.array([self.clientsObj[idx].test_datasize for idx in self.selected_clients_idx])
        self.receive_client_weight = self.receive_client_datasize / self.receive_client_datasize.sum()
        self.receive_client_next_models = [self.clientsObj[idx].next_step_model for idx in self.selected_clients_idx]


    def flatten(self, source):
        """
        flatten a dictionary
        :return: a flatten tensor
        :rtype:
        """
        return torch.cat([value.flatten() for name, value in source.items()])

    def column_normalization(self, matrix):
        """
        For a real matrix, sum over columns and then normalize each column to [0,1]
        :param matrix:
        :type matrix: numpy
        :return: normalized matrix
        :rtype:
        """
        if matrix.ndim == 1:
            column_sum = np.sum(matrix)
            return matrix / column_sum
        elif matrix.ndim == 2:
            result = np.zeros_like(matrix)
            M, N = matrix.shape
            for n in range(N):
                column_sum = np.sum(matrix[:, n])
                result[:, n] = matrix[:, n] / column_sum
            return result
        else:
            print("The input tensor array is not 1- or 2-dimensional.")
            return None

    def prune_matrix(self, matrix, threshold=0.01):
        """inspect matrix and set small values directly to zero"""
        mask = matrix < threshold
        matrix[mask] = 0.0
        return matrix

    def column_top_k(self, matrix, K=5):
        """inspect matrix and only store the top-K weight for each column"""
        total_num = matrix.shape[0]
        omit_num = (total_num - K) if (total_num > K) else 0
        for col in range(matrix.shape[1]):
            weights = matrix[:, col]
            mask = np.argpartition(weights, omit_num)[0:omit_num]
            weights[mask] = 0
            matrix[:, col] = weights
        return matrix


    def cal_norm(self, model1, model2):

        for idx, (k,v) in enumerate(model1.items()):
            if idx == 0:
                sum = torch.norm(v-model2[k])**2
            else:
                sum += torch.norm(v-model2[k])**2
        sum = 1 / sum
        return sum.detach().cpu().numpy()


    def cal_optimal_weight(self):
        """
        calcluate the optimal weight according to the formula
        return a left stochastic matrix where the k column is the neighbor weight for client k
        :return: a numpy matrix
        """
        weight_matrix = np.zeros([len(self.selected_clients_idx), len(self.selected_clients_idx)], dtype=np.float32)
        # server_send_model = [self.send_client_models[i] for i in range(len(self.selected_clients_idx))]
        """
        implementation method 1
        """
        # for col, state1 in enumerate(self.receive_client_models):
        #     for row, state2 in enumerate(self.receive_client_models):
        #         s1_col_trainafter = self.flatten(state1)
        #         s1_col_trainbefore = self.flatten(server_send_model[col])
        #         s2_row_trainafter = self.flatten(state2)
        #         weight_matrix[row, col] = np.array(torch.norm(2 * s1_col_trainafter - s1_col_trainbefore - s2_row_trainafter).cpu()**(-2))
        """
        implementation method 2
        """
        # flatten1_models = [self.flatten(state1) for state1 in self.receive_client_next_models]
        # flatten2_models = [self.flatten(state2) for state2 in self.receive_client_models]
        # for col, s1_col_trainafter in enumerate(flatten1_models):
        #     for row, s2_row_trainafter in enumerate(flatten2_models):
        #         weight_matrix[row, col] = (torch.norm(s1_col_trainafter - s2_row_trainafter).cpu()**(-2)).detach().cpu().numpy()

        """
        a new implementation for method 2
        """
        for col, s1_col_trainafter in enumerate(self.receive_client_next_models):
            for row, s2_row_trainafter in enumerate(self.receive_client_models):
                weight_matrix[row, col] = self.cal_norm(s1_col_trainafter, s2_row_trainafter)


        weight_matrix = self.column_normalization(weight_matrix)
        threshold = 1 / self.num_clients
        # prune the matrix
        # weight_matrix = self.prune_matrix(weight_matrix, threshold=threshold)
        # select top K
        weight_matrix = self.column_top_k(weight_matrix,self.feddwa_topk)
        # normalization again
        weight_matrix = self.column_normalization(weight_matrix)
        return weight_matrix

    def aggregated(self, optimal_matrix):
        """
                Aggregation method for FedDWA, use the optimal weight and the received model in
        the current round  to calculate the next round model which will be sent to each client
        the results are stored in self.send_client_models
        :param optimal_matrix:
        :type optimal_matrix: tensor matrix
        :return:
        :rtype:
        """
        assert (len(self.selected_clients_idx) > 0)
        # send_client_models = copy.deepcopy(self.receive_client_models)
        send_client_models = self.receive_client_next_models
        for idx, model_params in enumerate(send_client_models):
            weights = optimal_matrix[:, idx]
            for name in model_params:
                tmp = torch.stack([source[name].data.clone() * weights[row] for row, source in enumerate(self.receive_client_models)])
                tmp = torch.sum(tmp, dim=0).clone()
                model_params[name].data = tmp
        self.send_client_models = {idx: send_client_models[num] for num, idx in enumerate(self.selected_clients_idx)}

    def train(self):

        for i in range(self.global_rounds):
            self.selected_clients_idx = self.select_client()

            self.send_models()
            if i % self.evaluate_gap == 0:
                clients_test_acc, client_mean_test_acc = self.evaluate_acc()  # test all clients
                self.client_test_acc_logs.append(clients_test_acc)
                self.client_mean_test_acc_logs.append(client_mean_test_acc)

            t1 = time.time()
            loss_logs = []
            for idx in self.selected_clients_idx:
                client_loss = self.clientsObj[idx].train()
                loss_logs.append(client_loss)
            self.client_train_loss_logs.append(loss_logs)

            self.receive_models()
            optimal_weight = self.cal_optimal_weight()             # calculate the optimal weight

            self.aggregated(optimal_weight) # the new model parameters are stored in self.send_client_models
            t2 = time.time()

            self.logger.info(f'global round = {i + 1:d}, '
                  f'cost = {t2 - t1:.4f}s, '
                  f'clients mean acc = {np.array(client_mean_test_acc).mean():.4f}, ')
        self.save_results()



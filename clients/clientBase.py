import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class ClientBase(object):
    """
    Base class for clients in federated learning
    """
    def __init__(self, args, id, modelObj, train_set, test_set, **kwargs):
        super(ClientBase, self).__init__()
        self.device = args.device
        self.model = copy.deepcopy(modelObj)
        self.id = id
        self.train_set = train_set
        self.test_set = test_set
        self.batch_size = args.B
        self.data_size = len(train_set)
        self.test_datasize = len(test_set)
        self.lr = args.lr
        self.E = args.E
        self.optimizer = None
        self.loss_fn = None

        # at each communication round, store the init model from server
        self.W_old = {key: torch.zeros_like(value).to(self.device) for key, value in self.model.named_parameters()}
        # at each round, store the variation of model parameters
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def set_optim(self, optim, init_optim=True):
        """
         Set the optimizer that this model will perform SGD with.

         Args:
             - optim:     (torch.optim) that model will perform SGD steps with
             - init_optim (bool)        whether to initialise optimiser params
         """
        self.optimizer = optim

    def receive_models(self, new_model):
        """receive model from server and store the model in W_old as a dictionary"""
        # load the model
        for new_param, old_param in zip(new_model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        # global_model = new_model.state_dict()
        # self.model.load_state_dict(global_model)

        model_parameter = {key: value for key, value in self.model.named_parameters()}
        self.W_old = copy.deepcopy(model_parameter)

    def clone_model(self, source, target):
        """clone the source model parameter to the target parameter"""
        target_model = target.state_dict()
        source_model = source.state_dict()
        for k,v in source_model.items():
            target_model[k] = v
        target.load_state_dict(target_model)

    def test_accuracy(self):
        """compute accuracy using test set"""
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        # save acc
        return acc

    def cal_model_variation_(self, target, subtrahend):
        """
        target = current model parameter - subtrahend, noted that they are all dictionary with tensor value
        """
        current_model = {key: value for key, value in self.model.named_parameters()}
        for name in target:
            target[name].data = current_model[name].data.clone() - subtrahend[name].data.clone()

    def train_one_step(self,x,y):
        """
        Perform one step of SGD using assigned optimizer
        Args:
            - x: (torch.tensor) inputs
            - y: (torch.tensor) targets

        Returns:
            tupe of floats (loss, gradient) calculated during the training step.
        """
        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        loss.backward()
        # calculate gradient
        client_gradient = [torch.zeros_like(value) for key, value in self.model.named_parameters()]  # list
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                client_gradient[idx] += y.shape[0] * p.grad.data.clone()
        self.optimizer.step()

        return loss.item(), client_gradient





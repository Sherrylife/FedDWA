import torch
import torch.nn as nn
import numpy as np
import time
import copy
from clients.clientBase import ClientBase


class clientFedDWA(ClientBase):

    def __init__(self,args, id, modelObj, train_set, test_set, **kwargs):
        super(clientFedDWA, self).__init__(args, id, modelObj, train_set, test_set, **kwargs)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.0)
        self.next_step_model = None
        self.next_round = args.next_round


    def train_one_step(self,):
        """
        train one step using the dataset(x,y) to obtain the new model parameter,
        but we don't replace the self.model by the new model parameter, we only want
        to calculate the new model parameter.
        """
        # save the old model parameter
        old_model = copy.deepcopy(self.model.state_dict())

        self.model.train()
        for e in range(self.E):
            for data in self.train_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # self.person_optimizer.zero_grad()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
        self.next_step_model = {key: copy.deepcopy(value) for key, value in self.model.named_parameters()}
        # restore the old model
        self.model.load_state_dict(old_model)

    def test_accuracy(self):
        """
        Rewrite the method in clientBase, since in the method, for each client,
        they have their personalized model, and we use the personalized model
        to test the data.
        """
        correct = 0
        total = 0
        old_model = copy.deepcopy(self.model.state_dict())
        if self.next_step_model is not None:
            cur_model = self.model.state_dict()
            for k, v in self.next_step_model.items():
                cur_model[k] = v
            self.model.load_state_dict(cur_model)

        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                _, predicts = torch.max(outputs, 1)
                correct += (predicts == labels).sum().item()
                total += len(labels)
        acc = correct / total
        self.model.load_state_dict(old_model)
        return acc


    def receive_models(self, new_state):
        """
        Rewrite the receive_models method, because the new_state is come from
        model.named_parameters()  instead of model.state_dict()
        """
        current_state = self.model.state_dict()
        for name,value in new_state.items():
            current_state[name] = value
        self.model.load_state_dict(current_state)

    def train(self):
        start_time = time.time()
        loss_logs = []
        self.model.train()
        for e in range(self.E):
            for data in self.train_loader:
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # forward + backward
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                # optimize
                self.optimizer.step()
                # todo: save loss
                loss_logs.append(loss.mean().item())
        # get the model parameters of the next round by training in advance
        for i in range(self.next_round):
            self.train_one_step()
        self.optimizer.zero_grad(set_to_none=True)


        end_time = time.time()
        return np.array(loss_logs).mean()

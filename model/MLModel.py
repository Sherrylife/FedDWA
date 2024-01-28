# Several basic machine learning models
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import copy
from torchvision import models

class LogisticRegression(nn.Module):
    """A simple implementation of Logistic regression model"""

    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()

        self.num_feature = num_feature
        self.output_size = output_size
        self.linear = nn.Linear(self.num_feature, self.output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.linear(x)


class MLP(nn.Module):
    """A simple implementation of Deep Neural Network model"""

    def __init__(self, num_feature, output_size):
        super(MLP, self).__init__()
        self.hidden = 200
        self.model = nn.Sequential(
            nn.Linear(num_feature, self.hidden),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.hidden, output_size))

    def forward(self, x):
        return self.model(x)


class MlpModel(nn.Module):
    """
    2-hidden-layer fully connected model, 2 hidden layers with 200 units and a
    BN layer. Categorical Cross Entropy loss.
    """
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        """
        Returns a new MNISTModelBN.
        """
        super(MlpModel, self).__init__()
        self.in_features = in_features
        self.fc0 = torch.nn.Linear(in_features, hidden_dim)
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(hidden_dim, 200)
        self.relu1 = torch.nn.ReLU()
        self.out = torch.nn.Linear(200, num_classes)
        self.bn0 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn_layers = [self.bn0]

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, self.in_features)
        a = self.bn0(self.relu0(self.fc0(x)))
        b = self.relu1(self.fc1(a))

        return self.out(b)


class MnistCNN(nn.Module):
    """from fy"""
    def __init__(self, data_in, data_out):
        super(MnistCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
class FedAvgMLP(nn.Module):
    def __init__(self, in_features=784, num_classes=10, hidden_dim=200):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


# https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/cnn.py
class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features,
                               32,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(32,
                               64,
                               kernel_size=5,
                               padding=0,
                               stride=1,
                               bias=True)
        self.fc1 = nn.Linear(dim, 512)
        self.fc = nn.Linear(512, num_classes)

        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.maxpool(x)
        x = self.act(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc(x)
        return x


"""from fy"""
class CifarCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(CifarCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 384),
            nn.ReLU(),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = x.view(-1, 64 * 4 * 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CifarCNN_MTFL(nn.Module):
    """
    cifar10 model of MTFL
    """

    def __init__(self, data_in, data_out):
        super(CifarCNN_MTFL, self).__init__()

        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 10)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]

    def forward(self, x):
        """
        Returns outputs of model given data x.
        Args:
            - x: (torch.tensor) must be on same device as model
        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.fc0(self.flat(b)))

        return self.out(c)


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class BasicCNN(nn.Module):
    def __init__(self, data_in, data_out):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.fc(x)
        return x

"""Cluster FL"""
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 4 * 4, 62)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.fc1(x)
        return x


"""FedFomo"""
class BaseConvNet(nn.Module):
    def __init__(self, in_features=1, num_classes=10, ):
        super(BaseConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNMnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(data_in, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, data_out)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNFashion_Mnist(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


"""
Communication-Efficient Learning of Deep Networks from Decentralized Data
https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
"""
class CNNCifar(nn.Module):
    def __init__(self, data_in, data_out):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, data_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# TPDS MTFL model
class CIFAR10Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR10Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 10)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]


        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)

# TPDS MTFL model
class CIFAR100Model(nn.Module):
    def __init__(self, in_features, num_classes):
        super(CIFAR100Model, self).__init__()
        self.conv0 = torch.nn.Conv2d(3, 32, 3, 1)
        self.relu0 = torch.nn.ReLU()
        self.pool0 = torch.nn.MaxPool2d(2, 2)

        self.conv1 = torch.nn.Conv2d(32, 64, 3, 1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.flat = torch.nn.Flatten()
        self.fc0 = torch.nn.Linear(2304, 512)
        self.relu2 = torch.nn.ReLU()

        self.out = torch.nn.Linear(512, 100)

        self.drop = torch.nn.Dropout(p=0.5)

        self.bn0 = torch.nn.BatchNorm2d(32)
        self.bn1 = torch.nn.BatchNorm2d(64)

        # self.bn_layers = [self.bn0, self.bn1]
        self.classifier_layer = [self.fc0, self.out]
        self.head = [self.out]
        self.body = [self.conv0,self.conv1,self.bn0, self.bn1,self.fc0]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        a = self.bn0(self.pool0(self.relu0(self.conv0(x))))
        b = self.bn1(self.pool1(self.relu1(self.conv1(a))))
        c = self.relu2(self.drop(self.fc0(self.flat(b))))
        return self.out(c)



# from TPDS
class FashionMNISTModel(nn.Module):
    def __init__(self, num_classes):
        """
        Returns a new FashionMNISTModel.

        Args:
            - device: (torch.device) to place model on
        """
        super(FashionMNISTModel, self).__init__()
        self.conv0 = torch.nn.Conv2d(1, 32, 7, padding=3)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.bn0 = torch.nn.BatchNorm2d(32)
        self.conv1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.out = torch.nn.Linear(64 * 7 * 7, num_classes)
        self.bn_layers = [self.bn0, self.bn1]
        self.head = [self.out]
        self.body = [self.conv0,self.bn0,self.conv1,self.bn1]

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        """
        Returns outputs of model given data x.

        Args:
            - x: (torch.tensor) must be on same device as model

        Returns:
            torch.tensor model outputs, shape (batch_size, 10)
        """
        x = x.reshape(-1, 1, 28, 28)
        x = self.bn0(self.pool(self.act(self.conv0(x))))
        x = self.bn1(self.pool(self.act(self.conv1(x))))
        x = x.flatten(1)
        return self.out(x)


class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.relu = torch.nn.ReLU()
        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)
        self.classifier_layer = [self.fc1, self.output]
        self.head = [self.output]
        self.body = [self.conv1,self.conv2,self.fc1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class Reswithoutcon(nn.Module):
    def __init__(self, option='resnet50', pret=False, with_con=True, num_classes=10):
        super(Reswithoutcon, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet34':
            model_ft = models.resnet34(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret,num_classes=num_classes,zero_init_residual=True)
        
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x


class MobilenetV2(nn.Module):
    def __init__(self, option='MobilenetV2', pret=False, with_con=True,num_classes=10):
        super(MobilenetV2, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        model_ft = models.mobilenet_v2(pretrained=pret,num_classes=num_classes)
        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            self.body = temp
            self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
            
    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        vals.append(copy.deepcopy(temp.weight))
                    if hasattr(temp, 'bias'):
                        vals.append(copy.deepcopy(temp.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                for temp in bn:
                    if hasattr(temp, 'weight'):
                        temp.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(temp, 'bias'):
                        temp.bias.copy_(vals[i])
                        i = i + 1

    def forward(self, x):
        # x = self.features(x)
        if self.with_con:
            x = self.features(x)
            return x
        else:
            x = self.class_fit(x)
            return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=200):
        super(ResNet18, self).__init__()

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        # self.fc = nn.Linear(512, num_classes).to(device)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # self.bn_layers = [self.bn0, self.bn1]
        # self.linear_layers = [self.fc0,self.out]
        # self.deep = [self.bn0, self.bn1,self.out]
        # self.shallow = [self.conv0,self.conv1,self.fc0]
        self.head = [self.fc]
        self.body = [self.layer1, self.layer2, self.layer3, self.layer4]

    # 这个函数主要是用来，重复同一个残差块
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # 在这里，整个ResNet18的结构就很清晰了
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        # print(out.shape)
        out = self.fc(out)
        # print(out)
        return out

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                vals.append(copy.deepcopy(bn.weight))
                vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                bn.weight.copy_(vals[i])
                bn.bias.copy_(vals[i+1])
                i = i + 2

    def calc_acc(self, logits, y):
        """
        Calculate top-1 accuracy of model.

        Args:
            - logits: (torch.tensor) unnormalised predictions of y
            - y:      (torch.tensor) true values

        Returns:
            torch.tensor containing scalar value.
        """
        return (torch.argmax(logits, dim=1) == y).float().mean()

    def empty_step(self):
        """
        Perform one step of SGD with all-0 inputs and targets to initialise
        optimiser parameters.
        """
        # self.train_step(torch.zeros((2, 3, 64, 64),
        #                             device=self.device,
        #                             dtype=torch.float32),
        #                 torch.zeros((2),
        #                             device=self.device,
        #                             dtype=torch.int32).long())
        pass


def get_mobilenet(num_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param num_classes:
    :return: nn.Module
    """
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model

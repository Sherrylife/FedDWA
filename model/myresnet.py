'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
### For CIFAR10
### https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import copy
from collections.abc import Iterable

from torch.autograd import Variable

# __all__ = ['Resnetwithoutcon_','ResNet', 'resnet8','resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202','Reswithoutcon_']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.head = [self.linear]
        self.body = [self.conv1,self.bn1,self.layer1,self.layer2,self.layer3]

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x,output_features=False,with_con=True):
        if with_con:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.maxpool(out)
            # print(out.size())
            # 1/0
            features = out
        else:
            out = x
            features = out
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if output_features:
            return out, features
        else:
            return out

class ResNet8(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet8, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(16, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,option='A'))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x,output_features=False,with_con=True):
        if with_con:
            out = F.relu(self.bn1(self.conv1(x)))
            out = F.maxpool
            # print(out.size())
            # 1/0
            features = out
        else:
            out = x
            features = out
        out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if output_features:
            return out, features
        else:
            return out


# def resnet8(num_classes=10):
#     return ResNet8(BasicBlock, [3],num_classes=num_classes)
def resnet8(num_classes=10):
    return ResNet(BasicBlock, [1,1,1], num_classes=num_classes)

def resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3],num_classes=num_classes)


def resnet32(num_classes=10):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)


# def resnet56(num_classes=10):
#     return ResNet(BasicBlock, [9, 9, 9],num_classes=num_classes)
def resnet56(num_classes=10):
    return ResNet(BasicBlock, [6, 6, 6], num_classes=num_classes)


def resnet110(num_classes=10):
    return ResNet(BasicBlock, [18, 18, 18],num_classes=num_classes)


def resnet1202(num_classes=10):
    return ResNet(BasicBlock, [200, 200, 200],num_classes=num_classes)


class Resnetwithoutcon_(nn.Module):
    def __init__(self, option='resnet56', with_con=True,output_features=False,num_classes=10):
        super(Resnetwithoutcon_, self).__init__()
        self.dim = 2048
        self.with_con = with_con
        if option == 'resnet8':
            model_ft = resnet8(num_classes=num_classes)
            self.dim = 512
        if option == 'resnet20':
            model_ft = resnet20(num_classes=num_classes)
            self.dim = 512
        if option == 'resnet32':
            model_ft = resnet32(num_classes=num_classes)
        if option == 'resnet56':
            model_ft = resnet56(num_classes=num_classes)
        if option == 'resnet110':
            model_ft = resnet110(num_classes=num_classes)
        if option == 'resnet1202':
            model_ft = resnet1202(num_classes=num_classes)
        self.output_features = output_features
        # if with_con:
        #     self.model = model_ft
        # else:
        #     mod = list(model_ft.children())
        #     # print(mod)
        #     mod.pop(0)
        #     # print('--------------------zzzzzzzzzzzzzzzzzzzzz------------------')
        #     # print(mod)
        #     # 1/0
        #     self.model = nn.Sequential(*mod)
        self.model = model_ft

        mod = list(model_ft.children())
        if with_con:
            temp = mod.pop(0)
            self.features = model_ft
            # self.body = temp
            # self.head = mod
        else:
            mod = list(model_ft.children())
            mod.pop(0)
            self.class_fit = nn.Sequential(*mod)
        self.body = model_ft.body
        self.head = model_ft.head

    def get_head_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.head:
                if isinstance(bn, Iterable):
                    for temp in bn:
                        if hasattr(temp, 'weight') and temp.weight is not None:
                            vals.append(copy.deepcopy(temp.weight))
                        if hasattr(temp, 'bias') and temp.bias is not None:
                            vals.append(copy.deepcopy(temp.bias))
                else:
                    if hasattr(bn, 'weight') and bn.weight is not None:
                        vals.append(copy.deepcopy(bn.weight))
                    if hasattr(bn, 'bias') and bn.bias is not None:
                        vals.append(copy.deepcopy(bn.bias))
        return vals
    
    def get_body_val(self):
        vals = []
        with torch.no_grad():
            for bn in self.body:
                if isinstance(bn, Iterable):
                    for temp in bn:
                        if hasattr(temp, 'weight') and temp.weight is not None:
                            vals.append(copy.deepcopy(temp.weight))
                        if hasattr(temp, 'bias') and temp.bias is not None:
                            vals.append(copy.deepcopy(temp.bias))
                else:
                    if hasattr(bn, 'weight') and bn.weight is not None:
                        vals.append(copy.deepcopy(bn.weight))
                    if hasattr(bn, 'bias') and bn.bias is not None:
                        vals.append(copy.deepcopy(bn.bias))
        return vals

    def set_head_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.head:
                if isinstance(bn, Iterable):
                    for temp in bn:
                        if hasattr(temp, 'weight') and temp.weight is not None:
                            temp.weight.copy_(vals[i])
                            i = i + 1
                        if hasattr(temp, 'bias') and temp.bias is not None:
                            temp.bias.copy_(vals[i])
                            i = i + 1
                else:
                    if hasattr(bn, 'weight') and bn.weight is not None:
                        bn.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(bn, 'bias') and bn.bias is not None:
                        bn.bias.copy_(vals[i])
                        i = i + 1  

    def set_body_val(self,vals):
        i = 0
        with torch.no_grad():
            for bn in self.body:
                if isinstance(bn, Iterable):
                    for temp in bn:
                        if hasattr(temp, 'weight') and temp.weight is not None:
                            temp.weight.copy_(vals[i])
                            i = i + 1
                        if hasattr(temp, 'bias') and temp.weight is not None:
                            temp.bias.copy_(vals[i])
                            i = i + 1
                else:
                    if hasattr(bn, 'weight') and bn.weight is not None:
                        bn.weight.copy_(vals[i])
                        i = i + 1
                    if hasattr(bn, 'bias') and bn.bias is not None:
                        bn.bias.copy_(vals[i])
                        i = i + 1  

    def forward(self, x,output_features=False,with_con=True):
        # x = self.features(x)
        # if self.with_con:
        #     if output_features:
        #         x, features = self.model(x,output_features=output_features)
        #         return x , features
        #     else:
        #         x = self.model(x,output_features=output_features)
        #         return x
        # else:
        #     if output_features:
        #         x, features = self.model(x,output_features=output_features)
        #         return x , features
        #     else:
        #         x = self.model(x,output_features=output_features)
        #         return x
        # self.with_con = with_con
        if output_features:
            x, features = self.model(x,output_features=output_features,with_con=with_con)
            return x , features
        else:
            x = self.model(x,output_features=output_features,with_con=with_con)
            return x

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
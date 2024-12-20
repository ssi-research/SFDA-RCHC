"""
 This file was copied from https://github.com/tim-learn/SHOT-plus/code/uda/network.py and modified for this project needs.
 The license of the file is in: https://github.com/tim-learn/SHOT-plus/blob/master/LICENSE
"""


import torch.nn as nn
import torch
from torchvision import models
import convnext

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ConvnextTiny(nn.Module):
    def __init__(self):
        super(ConvnextTiny, self).__init__()
        self.model_convnext = convnext.convnext_tiny(pretrained=True, in_22k=False)
        self.in_features = 768

    def forward(self, x):
        x = self.model_convnext.forward_features(x)
        x = x.view(x.size(0), -1)
        return x


class Identity(nn.Module):

    def __init__(self, sub=1.0):
        super(Identity, self).__init__()
        self.sub=sub

    def forward(self,x):
        return x*self.sub


class ConvnextTiny2(nn.Module):
    def __init__(self, in_22k=False):
        super(ConvnextTiny2, self).__init__()
        self.model_convnext = convnext.convnext_tiny(pretrained=True, in_22k=in_22k)
        # self.model_convnext.fc=Identity()
        self.in_features = 768

    def forward(self, x):
        x = self.model_convnext.forward_features(x)
        # x = self.model_convnext(x)
        x = x.view(x.size(0), -1)
        return x


class ConvnextSmall(nn.Module):
    def __init__(self, in_22k=False):
        super(ConvnextSmall, self).__init__()
        self.model_convnext = convnext.convnext_small(pretrained=True, in_22k=in_22k)
        self.in_features = 768

    def forward(self, x):
        x = self.model_convnext.forward_features(x)
        # x = self.model_convnext(x)
        x = x.view(x.size(0), -1)
        return x


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn" or self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.bn(x)
        if self.type == "bn_relu" or self.type == "bn_relu_drop":
            x = self.relu(x)
        if self.type == "bn_relu_drop":
            x = self.dropout(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = torch.nn.utils.weight_norm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        elif type == 'linear':
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        if not self.type in {'wn', 'linear'}:
            w = self.fc.weight
            w = torch.nn.functional.normalize(w, dim=1, p=2)

            x = torch.nn.functional.normalize(x, dim=1, p=2)
            x = torch.nn.functional.linear(x, w)
        else:
            x = self.fc(x)
        return x

class netSHOT(nn.Module):

    def __init__(self, num_C, cnn, E_dims, apply_wn=False, type_bottleneck="bn", pretrained_path=None):

        super(netSHOT, self).__init__()

        # Frozen initial conv layers
        if cnn == 'resnet50' or cnn == 'resnet101':
            self.M = ResBase(res_name=cnn)
            feature_dim=2048
        elif cnn == 'convnextTiny22':
            self.M = ConvnextTiny2(in_22k=True)
            feature_dim = 768
        elif cnn == 'convnextSmall22':
            self.M = ConvnextSmall(in_22k=True)
            feature_dim = 768
        else:
            raise NotImplementedError('Not implemented for ' + str(cnn))

        self.E = feat_bootleneck(feature_dim=feature_dim, bottleneck_dim=E_dims, type=type_bottleneck)

        if apply_wn:
            self.G = feat_classifier(num_C, E_dims, type="wn")
        else:
            self.G = feat_classifier(num_C, E_dims, type="linear")

        self.components = {
            'M': self.M,
            'E': self.E,
            'G': self.G,
        }

    def forward(self, x, which_fext='original'):
        raise NotImplementedError('Implemented a custom forward in train loop')

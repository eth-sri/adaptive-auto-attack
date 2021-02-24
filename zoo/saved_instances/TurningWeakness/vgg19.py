import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to('cuda')
        std = torch.tensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to('cuda')
        x = (x - mean) / std
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M']))


def random_label(label):
    class_num = 10
    attack_label = np.random.randint(class_num)
    while label == attack_label:
        attack_label = np.random.randint(class_num)
    return attack_label


def l1_detection(model, x, n_radius):
    return torch.norm(F.softmax(model(x), dim=1) - F.softmax(model(x + n_radius * torch.randn_like(x)), dim=1), 1).item()


def untargeted_detection(model, x, lr, u_radius, thres,
                         margin=20, use_margin=True):
    model.eval()
    x_var = torch.autograd.Variable(x.clone().cuda(), requires_grad=True)
    # x_var = x.clone()
    # x_var.requires_grad_()
    true_label = model(x_var).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = torch.optim.SGD([x_var], lr=lr)
    counter = 0
    with torch.enable_grad():
        while model(x_var).data.max(1, keepdim=True)[1][0].item() == true_label:
            optimizer_s.zero_grad()
            output = model(x_var)
            if use_margin:
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == true_label:
                    argmax11 = top2_1[0][1]
                loss = (output[0][true_label] - output[0][argmax11] + margin).clamp(min=0)
            else:
                loss = -F.cross_entropy(output, torch.LongTensor([true_label]).cuda())
            loss.backward()

            x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - x, min=-u_radius, max=u_radius) + x
            counter += 1
            if counter > thres:
                return False
    return True


def targeted_detection(model, x, lr,
                         t_radius,
                         thres,
                         margin=20,
                         use_margin=False):
    model.eval()
    x_var = torch.autograd.Variable(x.clone().cuda(), requires_grad=True)
    true_label = model(x_var).data.max(1, keepdim=True)[1][0].item()
    optimizer_s = torch.optim.SGD([x_var], lr=lr)
    counter = 0
    with torch.enable_grad():
        while model(x_var).data.max(1, keepdim=True)[1][0].item() == true_label:
            optimizer_s.zero_grad()
            target_l = torch.LongTensor([random_label(true_label)]).cuda()
            output = model(x_var)
            if use_margin:
                target_l = target_l[0].item()
                _, top2_1 = output.data.cpu().topk(2)
                argmax11 = top2_1[0][0]
                if argmax11 == target_l:
                    argmax11 = top2_1[0][1]
                loss = (output[0][argmax11] - output[0][target_l] + margin).clamp(min=0)
            else:
                loss = F.cross_entropy(output, target_l)
            loss.backward()

            x_var.data = torch.clamp(x_var - lr * x_var.grad.data, min=0, max=1)
            x_var.data = torch.clamp(x_var - x, min=-t_radius, max=t_radius) + x
            counter += 1
            if counter > thres:
                return False
    return True


def l1_vals(model, x, thres, n_radius):
    vals = np.zeros(0)
    N = len(x)
    for i in range(N):
        adv = x[i, :]
        val = l1_detection(model, adv, n_radius)
        vals = np.concatenate((vals, [val]))
    return vals < thres


def untargeted_vals(model, x, thres, lr, u_radius):
    vals = np.zeros(0)
    N = len(x)
    for i in range(N):
        adv = x[i, :]
        model.eval()
        val = untargeted_detection(model, adv, lr, u_radius, thres)
        vals = np.concatenate((vals, [val]))
    return vals


def targeted_vals(model, x, thres, lr, t_radius):
    vals = np.zeros(0)
    N = len(x)
    for i in range(N):
        adv = x[i, :]
        val = targeted_detection(model, adv, lr, t_radius, thres)
        vals = np.concatenate((vals, [val]))
    return vals


class TurningWeaknessDetector(nn.Module):
    def __init__(self, classifier):
        super(TurningWeaknessDetector, self).__init__()
        self.classifier = classifier
        self.threshold_C1 = 0.006
        self.threshold_C2t = 10
        self.threshold_C2u = 10
        self.lr_u = 0.0025
        self.lr_t = 0.0025
        self.radius_u = 8/255
        self.radius_t = 8/255
        self.radius_n = 0.01

    def forward(self, x):
        result_u = untargeted_vals(self.classifier, x, self.threshold_C2u, self.lr_u, self.radius_u)
        result_t = targeted_vals(self.classifier, x, self.threshold_C2t, self.lr_t, self.radius_t)
        result_l1 = l1_vals(self.classifier, x, self.threshold_C1, self.radius_n)
        result = np.logical_and(np.logical_and(result_u, result_t), result_l1)
        return torch.tensor(result)

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.nn import functional as F
import numpy as np
import torchvision.transforms as transforms
from utils.scloss import SupConLoss

def rotate_img(img, s):
    transform = transforms.RandomResizedCrop(size=(32, 32), scale=(0.66, 0.67), ratio = (0.99,1.00))
    img = transform(img)
    return torch.rot90(img, s, [-1, -2])

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via SGD.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Sgd(ContinualModel):
    NAME = 'sgd'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Sgd, self).__init__(backbone, loss, args, transform)


    def observe(self, inputs, labels, not_aug_inputs):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()

        return loss.item()

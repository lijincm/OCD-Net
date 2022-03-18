
"""
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from copy import deepcopy
from torch.nn import functional as F
import numpy as np
import torchvision.transforms as transforms
from utils.scloss import SupConLoss
from utils.crd import CRDLoss

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via online contrastive distillation')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--ER_weight', type=float, default=1.0)
    parser.add_argument('--Bernoulli_probability', type=float, default=0.70)
    return parser

def rotate_img(img, s):
    transform = transforms.RandomResizedCrop(size=(32, 32), scale=(0.66, 0.67), ratio = (0.99,1.00))
    img = transform(img)
    return torch.rot90(img, s, [-1, -2])

class OCDNet(ContinualModel):
    NAME = 'OCDNet'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']
    def __init__(self, backbone, loss, args, transform):
        super(OCDNet, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        # Initialize the teacher model
        self.teacher_model = deepcopy(self.net).to(self.device)
        # Set parameters for experience replay
        self.ER_weight = args.ER_weight
        # Set Bernoulli_probability
        self.Bernoulli_probability = args.Bernoulli_probability
        # Set parameters for the CRD-Module
        self.criterion = SupConLoss(temperature= 0.07)
        self.crd = CRDLoss(temperature= 0.07)
        self.model_iterations = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs, _, bat_inv= self.net(inputs, return_features=True)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            teacher_logits, _, tea_inv= self.teacher_model(buf_inputs, return_features=True)
            student_outputs, _, buf_inv = self.net(buf_inputs, return_features=True)
            with torch.no_grad():
                teacher_model_prob = F.softmax(teacher_logits, 1)
                label_mask = F.one_hot(buf_labels, num_classes=teacher_logits.shape[-1]) > 0
                adaptive_weight = teacher_model_prob[label_mask]
            squared_losses = adaptive_weight * torch.mean((student_outputs - teacher_logits.detach()) ** 2 , dim=1)

            loss += 0.1 *  squared_losses.mean()
            loss += 0.1 * self.crd(buf_inv, tea_inv.detach(), buf_labels)
            loss += self.ER_weight * self.loss(student_outputs, buf_labels)

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            bat_inv = torch.cat((bat_inv, buf_inv))

        """Collaborative Contrastive Learning https://arxiv.org/pdf/2109.02426.pdf """
        label_shot = torch.arange(0, 4).repeat(inputs.shape[0])
        label_shot = label_shot.type(torch.LongTensor)
        choice = np.random.choice(a=inputs.shape[0], size=inputs.shape[0], replace=False)
        rot_label = label_shot[choice].to(self.device)
        rot_inputs = inputs.cpu()
        for i in range(0, inputs.shape[0]):
            rot_inputs[i] = rotate_img(rot_inputs[i], rot_label[i])
        rot_inputs = rot_inputs.to(self.device)
        _, rot_outputs, t_inv = self.net(rot_inputs, return_features=True)
        loss += 0.3 * self.criterion(bat_inv, t_inv, labels)
        loss += 0.3 * self.loss(rot_outputs, rot_label)

        loss.backward()
        self.opt.step()
        self.model_iterations += 1
        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:not_aug_inputs.shape[0]])

        #Updating the teacher model
        if torch.rand(1) < self.Bernoulli_probability:
            # Momentum coefficient m
            m = min(1 - 1 / (self.model_iterations + 1), 0.999)
            for teacher_param, param in zip(self.teacher_model.parameters(), self.net.parameters()):
                teacher_param.data.mul_(m).add_(alpha=1 - m, other=param.data)
        return loss.item()


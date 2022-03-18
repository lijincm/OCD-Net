# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset
import sys


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False, eval_tea=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    current_model = model.net
    if eval_tea:
        current_model = model.teacher_model
    status = current_model.training
    current_model.eval()

    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            outputs = current_model(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            #Task incremntal learnning results
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100
                    if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    current_model.train(status)

    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    model.net.to(model.device)
    results, results_mask_classes = [], []
    model_stash = create_stash(model, args, dataset)

    tea_loggers = {}
    tea_results = {}
    tea_results_mask_classes = {}

    if hasattr(model, 'teacher_model'):
        tea_results['teacher_model'], tea_results_mask_classes['teacher_model'] = [], []

    if args.csv_log:
        #csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
        if hasattr(model, 'teacher_model'):
            #print(f'Creating Logger for the teacher model')
            tea_loggers['teacher_model'] = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    random_results_class, random_results_task = evaluate(model, dataset_copy, eval_tea=True)
    print(file=sys.stderr)
    for t in range(dataset.N_TASKS):
        model.net.train()
        train_loader, test_loader = dataset.get_data_loaders()
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        for epoch in range(args.n_epochs):
            for i, data in enumerate(train_loader):
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)
                not_aug_inputs = not_aug_inputs.to(model.device)
                loss = model.observe(inputs, labels, not_aug_inputs)
                progress_bar(i, len(train_loader), epoch, t, loss)
                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)
                model_stash['batch_idx'] = i + 1
            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0
        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])
        mean_acc = np.mean(accs, axis=1)
        #print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)

        model_stash['mean_accs'].append(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)
        if hasattr(model, 'teacher_model'):
            #print(f'Evaluating teacher_model')
            tea_accs = evaluate(model, dataset, eval_tea=True)
            tea_results['teacher_model'].append(tea_accs[0])
            tea_results_mask_classes['teacher_model'].append(tea_accs[1])
            tea_mean_acc = np.mean(tea_accs, axis=1)
            print_mean_accuracy(tea_mean_acc, t + 1, dataset.SETTING)

        if args.csv_log:
            tea_loggers['teacher_model'].log(tea_mean_acc)
    if args.csv_log:
        tea_loggers['teacher_model'].add_fwt(results, random_results_class, results_mask_classes, random_results_task)
        tea_loggers['teacher_model'].add_bwt(tea_results['teacher_model'], tea_results_mask_classes['teacher_model'])
        tea_loggers['teacher_model'].add_forgetting(tea_results['teacher_model'], tea_results_mask_classes['teacher_model'])

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        tea_loggers['teacher_model'].write(vars(args))

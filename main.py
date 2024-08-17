from __future__ import division
from __future__ import print_function

import copy
import os

import utils_this
from models_layers.ts2vec import TS2Vec
from utils.utils_centerloss import compute_center_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from thop import profile
from dagma.linear import DagmaLinear
from torch.utils.benchmark import timer
from utils.utils_mixup import mixup_data, MixUpLoss
import csv
import datetime
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from utils_this import *
from modules import DecomposeWHAR
import sklearn.metrics as metrics
import warnings
import sys

# sys.setrecursionlimit(10000)
warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser()

# Define model and dataset parameters
parser.add_argument('--model', default='DecomposeWHAR', type=str,
                    choices=['DecomposeWHAR'])
parser.add_argument('--dataset', default='opp_24_12', type=str,
                    help='Dataset name, e.g. opp_24_12, opp_60_30, realdisp_40_20, realdisp_100_50, skoda_right_78_39, skoda_right_196_98')
parser.add_argument('--test-user', type=int, default=-1,
                    help='ID of test user.')
parser.add_argument('--Scheduling_lambda', type=float, default=0.995,
                    help='Scheduling lambda.')
parser.add_argument('--seed', type=int, default=1, choices=[1, 42, 100, 255, 999],
                    help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')

args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(torch.cuda.is_available())

# Set model hyperparameters based on the dataset
if args.dataset == "opp_24_12":
    args.window_size = 24
    args.user_num = 4
    args.class_num = 17
    args.node_num = 5
    args.node_dim = 9
    args.time_slice_window = 6
    args.D = 64
    args.P = 8
    args.S = 4
    args.kernel_size_tcn = 3
    args.window_sample = 24
    args.patch_size = 6
    args.time_denpen_len = int(args.window_sample / args.patch_size)
    args.num_windows = 5
    args.conv_kernel = 3
    args.conv_out = 48
    args.lr = 0.001
    args.epochs = 80
    args.batch_size = 64
    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

# Other dataset configurations are similarly set...

args.datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
args.experiment = args.dataset + "_" + args.model + "_" + args.datetime

print(args)

# Set random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Function for training the model
def train(model, train_loader, test_loader, optimizer, scheduler, epoch, adjacent_matrix):
    model.train()
    all_features = []
    all_labels = []
    loss_train = []

    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    for batch_idx, (data, label) in enumerate(train_loader):
        if data.shape[0] == 1:
            continue

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        optimizer.zero_grad()

        # Center loss calculation (if applicable)
        if args.center_loss:
            centers = model.centers

        # MixUp data augmentation (if applicable)
        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, label, args.alpha)

        # Model-specific forward pass
        if args.model == 'DecomposeWHAR':
            feature, output, attention = model(data)

        # Calculate loss (MixUp or normal CrossEntropy)
        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(output, y_a_y_b_lam)
        else:
            loss = criterion(output, label)

        # Add center loss to the total loss (if applicable)
        if args.center_loss:
            center_loss = compute_center_loss(feature, centers, label)
            loss += args.beta * center_loss

        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())

        if args.mixup:
            criterion = criterion.get_old()

    scheduler.step()

    # Validation phase
    correct1 = 0
    size = 0
    predicts = []
    labelss = []
    loss_val = []
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1:
            continue

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)

        if args.model == 'DecomposeAndFuseWHAR':
            feature, output = model(data)

        test_loss = criterion(output, label)
        pred1 = output.data.max(1)[1]
        k = label.data.size()[0]
        correct1 += pred1.eq(label.data).cpu().sum()
        size += k
        labels = label.cpu().numpy()
        labelss += list(labels)
        pred1s = pred1.cpu().numpy()
        predicts += list(pred1s)
        loss_val.append(test_loss.data.item())

    print(f'Epoch: {epoch:04d}, '
          f'train_loss: {np.mean(loss_train):.6f}, '
          f'test_loss: {np.mean(loss_val):.6f}, '
          f'test_acc: {1. * correct1.float() / size:.6f}, '
          f'test_f1: {metrics.f1_score(labelss, predicts, average="macro"):.6f}, '
          f'time: {time.time() - t:.4f}s')

    return metrics.f1_score(labelss, predicts,
                            average='macro'), 1. * correct1.float() / size, all_features, all_labels, out_attention


# Initialize model based on configuration
def init_model():
    if args.model == 'DecomposeWHAR':
        model = DecomposeWHAR(num_sensor=args.node_num,
                              M=args.node_dim,
                              L=args.window_size,
                              D=args.D,
                              P=args.P,
                              S=args.S,
                              kernel_size=args.kernel_size_tcn,
                              r=1,
                              num_layers=2,
                              num_classes=args.class_num)

    if args.cuda:
        model.cuda()
        get_info_params(model)
        get_info_layers(model)

        input = torch.randn(64, 5, 24, 9).cuda()
        flops, params = profile(model, (input,))
        print(f'flops: {flops / 1e6:.2f} M, params: {params / 1e6:.2f} M')

    return model


# Main function for training and evaluation
def main(test_user):
    best_f1 = 0.0
    best_acc = 0.0

    model = init_model()
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
    lambda1 = lambda epoch: args.Scheduling_lambda ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    train_loader, test_loader = load_data(name=args.dataset, batch_size=args.batch_size, test_user=test_user)


    # Training loop
    for epoch in range(args.epochs):
        f1_test, acc_test, all_features, all_labels, output_attention = train(model, train_loader, test_loader,
                                                                              optimizer, scheduler, epoch, None)

        if f1_test > best_f1:
            best_f1 = f1_test
            best_acc = acc_test.item()

    print(f"------------------------------------------Best epoch:  Acc is {best_acc}, f1 is {best_f1}")

    # Record experiment results to a CSV file
    with open('record.csv', mode='a', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        row = [args.dataset, args.model, args.datetime, args.lr, best_acc, best_f1, test_user, args.seed, args.epochs,
               'mixup' if args.mixup else '',
               args.d_model if args.d_model != 128 else '',
               args.top_k if args.top_k != 4 else '',
               args.e_layers if args.e_layers != 4 else '']
        csv_writer.writerow(row)


if __name__ == '__main__':
    # Run experiments for all users if test_user == -1, otherwise run for a specific user
    if args.test_user == -1:
        if args.user_num == 1:
            main(0)
        else:
            for index in range(args.user_num):
                main(index)
    else:
        main(args.test_user)

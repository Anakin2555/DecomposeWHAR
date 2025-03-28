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

warnings.filterwarnings("ignore")

# Parse command-line arguments
parser = argparse.ArgumentParser()

# Define model and dataset parameters
parser.add_argument('--model', default='DecomposeWHAR', type=str,
                    choices=['DecomposeWHAR'])
parser.add_argument('--dataset', default='opp_24_12', type=str,
                    help='Dataset name, e.g. opp_24_12, opp_60_30, realdisp_40_20, realdisp_100_50, skoda_right_78_39, skoda_right_196_98')
parser.add_argument('--test-user', type=int, default=0,
                    help='ID of test user.')
parser.add_argument('--Scheduling_lambda', type=float, default=0.995,
                    help='Scheduling lambda.')
parser.add_argument('--seed', type=int, default=1, choices=[1, 42, 100, 255, 999],
                    help='Random seed.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument("--mixup", default=False, help="using data augmentation")
parser.add_argument('--center_loss',default=False)

args, unknown = parser.parse_known_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(torch.cuda.is_available())
PATH_save_models='./model'

args.conv_kernels, args.kernel_size = 64, 5
args.rnn_num_layers = 2
args.rnn_is_bidirectional = False
args.hidden_dim = 128
args.activation = "ReLU"

# mixup alpha
args.alpha=0.8

# center_loss beta
args.beta = 0.3

# Transformer
args.n_heads=6
args.e_layers=4
args.d_ff=128
args.output_attention=False
args.factor=1
args.d_model=128


args.num_tcn_layer = 2
args.num_a_layers = 1
args.num_m_layers = 1


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


elif args.dataset == "opp_60_30":
    args.window_size = 60
    args.user_num = 4
    args.class_num = 17
    args.node_num = 5
    args.node_dim = 9
    args.intervals = 10
    args.window = 6

    args.D = 64
    args.P = 20
    args.S = 10
    args.kernel_size_tcn = 3

    args.moving_window = [2, 3]
    args.stride = [1, 1]
    args.window_sample = 60
    args.patch_size = 15
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


elif args.dataset == "realdisp_40_20":

    args.D = 64
    args.P = 10
    args.S = 5
    args.kernel_size_tcn = 3

    args.time_slice_window = 20
    args.window_size = 40
    args.user_num = 10
    args.class_num = 33
    args.node_num = 9
    args.node_dim = 9

    args.window_sample = 40
    args.patch_size = 20
    args.time_denpen_len = int(args.window_sample / args.patch_size)

    # 原始学习率为0.0001
    args.lr = 0.001

    args.epochs = 60
    args.batch_size = 128

    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

elif args.dataset == "realdisp_100_50":

    args.window_size = 100
    args.user_num = 10
    args.class_num = 33
    args.node_num = 9
    args.node_dim = 9
    args.window_sample = 100
    args.patch_size = 50
    args.time_denpen_len = int(args.window_sample / args.patch_size)

    args.D = 64
    args.P = 50
    args.S = 25
    args.kernel_size_tcn = 3

    args.lr = 0.001
    args.epochs = 60
    args.batch_size = 128

    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

elif args.dataset == "skoda_right_78_39":
    args.window_size = 78
    args.user_num = 1
    args.class_num = 10
    args.node_num = 10
    args.node_dim = 3

    args.D = 64
    args.P = 26
    args.S = 13
    args.kernel_size_tcn = 3

    args.time_slice_window = 26
    args.window_sample = 78
    args.patch_size = 39
    args.time_denpen_len = int(args.window_sample / args.patch_size)

    args.lr = 0.0001
    args.epochs = 80
    args.batch_size = 64

    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128

elif args.dataset == "skoda_right_196_98":
    args.window_size = 196
    args.user_num = 1
    args.class_num = 10
    args.node_num = 10
    args.node_dim = 3

    args.D = 64
    args.P = 98
    args.S = 49
    args.kernel_size_tcn = 3

    args.window_sample = 196
    args.patch_size = 98
    args.time_denpen_len = int(args.window_sample / args.patch_size)

    args.lr = 0.0001
    args.epochs = 80
    args.batch_size = 64

    args.channel_dim = 32
    args.time_reduce_size = 8
    args.hid_dim = 128




args.datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
args.experiment = args.dataset + "_" + args.model + "_" + args.datetime

print(args)

# Set random seeds for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)





# Function for training the model
def train(model, train_loader, test_loader, optimizer, scheduler, epoch,  adjacent_matrix, pretrain_model):
    model = model
    train_loader = train_loader
    test_loader = test_loader
    optimizer = optimizer
    scheduler = scheduler
    all_features=[]
    all_labels=[]

    t = time.time()    
    if args.cuda:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()

    loss_train = []
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        if data.shape[0] == 1:
            continue
    
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)
        optimizer.zero_grad()

        if args.center_loss:
            centers = model.centers

        if args.mixup:
            data, y_a_y_b_lam = mixup_data(data, label, args.alpha)
        if args.model =='DecomposeWHAR':
            feature,output= model(data)

        if args.mixup:
            criterion = MixUpLoss(criterion)
            loss = criterion(output, y_a_y_b_lam)
        else:
            loss = criterion(output, label)


        if args.center_loss:
            center_loss = compute_center_loss(feature, centers, label)
            loss = loss + args.beta * center_loss

        loss.backward()
        optimizer.step()
        loss_train.append(loss.data.item())

        if args.mixup:
            criterion = criterion.get_old()


    scheduler.step()


    # test phase
    correct1 = 0
    size = 0
    predicts = []
    labelss = []  
    loss_val = []
    attention_matrix_for_label=[]
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1:
            continue
        
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(
            label, volatile=True)
        if args.model =='DecomposeWHAR':
            feature,output = model(data)


    

        test_loss = criterion(output, label)
        pred1 = output.data.max(1)[1] 
        k = label.data.size()[0]
        correct1 += pred1.eq(label.data).cpu().sum()
        size += k    
        labels = label.cpu().numpy()
        labelss = labelss + list(labels)
        pred1s = pred1.cpu().numpy()
        predicts = predicts + list(pred1s)
        
        loss_val.append(test_loss.data.item())

    print('Epoch: {:04d}'.format(epoch),
          'train_loss: {:.6f}'.format(np.mean(loss_train)),
          'test_loss: {:.6f}'.format(np.mean(loss_val)),
          'test_acc:{:.6f}'.format(1. * correct1.float() / size),
          'test_f1: {:.6f}'.format(metrics.f1_score(labelss, predicts, average='macro')),
          'time: {:.4f}s'.format(time.time() - t))

    return metrics.f1_score(labelss, predicts, average='macro'), 1. * correct1.float() / size, all_features,all_labels



# test phase for skoda dataset
def test(model, test_loader):
    correct1 = 0
    size = 0
    predicts = []
    labelss = []
    loss_test = []
    t = time.time()
    attention_matrix_for_label = []
    model.eval()
    for batch_idx, (data, label) in enumerate(test_loader):
        if data.shape[0] == 1:
            continue

        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data, volatile=True), Variable(
            label, volatile=True)
        if args.model == 'DecomposeWHAR':
            feature, output = model(data)

        if args.cuda:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            criterion = nn.CrossEntropyLoss()

        test_loss = criterion(output, label)
        pred1 = output.data.max(1)[1]
        k = label.data.size()[0]
        correct1 += pred1.eq(label.data).cpu().sum()
        size += k
        labels = label.cpu().numpy()
        labelss = labelss + list(labels)
        pred1s = pred1.cpu().numpy()
        predicts = predicts + list(pred1s)

        loss_test.append(test_loss.data.item())

    print(
          'test_loss: {:.6f}'.format(np.mean(loss_test)),
          'test_acc:{:.6f}'.format(1. * correct1.float() / size),
          'test_f1: {:.6f}'.format(metrics.f1_score(labelss, predicts, average='macro')),
          'time: {:.4f}s'.format(time.time() - t))

    return metrics.f1_score(labelss, predicts, average='macro'), 1. * correct1.float() / size



# Initialize model based on configuration
def init_model():
    if args.model == 'DecomposeWHAR':
        model = DecomposeWHAR(
                         num_sensor= args.node_num,
                         M=args.node_dim,
                         L=args.window_size,
                         D=args.D,
                         P=args.P,
                         S=args.S,
                         kernel_size=args.kernel_size_tcn,
                         r=1,
                         num_m_layers=args.num_m_layers,
                         num_a_layers=args.num_a_layers,
                         num_layers=args.num_tcn_layer,
                         num_classes=args.class_num)
    if args.cuda:
        model.cuda()
        # get_info_params(model)
        # get_info_layers(model)

        # input = torch.randn(64, 5, 24, 9).cuda()
        # flops, params = profile(model, (input,))
        # print(f'flops: {flops / 1e6:.2f} M, params: {params / 1e6:.2f} M')

    return model


# Main function for training and evaluation
def main(test_user):
    global PATH_save_models
    best_f1=0.0
    best_acc=0.0

    model=init_model()
    optimizer = optim.Adam(list(model.parameters()),lr=args.lr)
    lambda1 = lambda epoch: args.Scheduling_lambda ** epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    if 'skoda' in args.dataset:
        train_loader, val_loader, test_loader = load_data(name=args.dataset, batch_size=args.batch_size, test_user=test_user)
    else:
        train_loader, test_loader = load_data(name = args.dataset, batch_size=args.batch_size, test_user=test_user)

    if args.model == 'TS2VEC':
        pretrain_model = pretrain(args.dataset, test_user)
    else:
        pretrain_model = None

    # train model
    for epoch in range(args.epochs):
        if 'skoda' in args.dataset:
            f1_test, acc_test, all_features, all_labels = train(model, train_loader, val_loader, optimizer, scheduler,
                                                                epoch, None, pretrain_model)
        else:
            f1_test,acc_test,all_features,all_labels =train(model, train_loader, test_loader, optimizer, scheduler, epoch, None,pretrain_model)

        if f1_test >best_f1:
            best_f1 = f1_test
            best_acc= acc_test.item()
            best_features = all_features


            ### Save model ########
            if not os.path.exists(PATH_save_models):
                os.makedirs(PATH_save_models)
            # 只保存最好的模型
            model_save_path = os.path.join(PATH_save_models, f'{args.model}_teacher_{test_user}.pth')
            torch.save(model, model_save_path)


    print(f"------------------------------------------------------------------------------Best epoch:  Acc is {best_acc}, f1 is {best_f1} ")


    # skoda dataset test phase
    if 'skoda' in args.dataset:
        model=torch.load(model_save_path)
        f1_test, acc_test= test(model,test_loader)
        print(f"------------------------------------------------------------------------------Skoda Test Result:  Acc is {acc_test}, f1 is {f1_test} ")





    # record each experiment result
    with open('record.csv', mode='a', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        row=[args.dataset,args.model,args.datetime,args.lr,best_acc, best_f1,test_user,args.seed,args.epochs,
             args.alpha if args.mixup else '',
             args.d_model if args.d_model !=128 else '',
             args.top_k if args.top_k != 4 else '',
             args.e_layers if args.e_layers !=4 else '',
             args.beta if args.center_loss else ''
             ]
        csv_writer.writerow(row)
        file.close()



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

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import data
from utils.utils import paint


def load_data(name='opp_24_12', batch_size=64, test_user=0):
    
    if 'opp' in name:
        _dataset = data.Opportunity(name = name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
    
        invalid_feature = np.arange(0, 36)
        train_x  = np.delete( train_x, invalid_feature, axis = 2 )
        test_x  = np.delete( test_x, invalid_feature, axis = 2 )
    
        train_x = train_x[:,:,:45]
        test_x = test_x[:,:,:45]
    
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 9)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 9)
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])

    elif 'realdisp' in name:
        _dataset = data.Realdisp(name = name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)
        
        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 13)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 13)
        train_x = np.transpose(train_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])
        
        train_x = train_x[:, :, :, :9]
        test_x = test_x[:, :, :, :9]


    elif 'skoda' in name:
        _dataset = data.Skoda(name = name)
        train_x, train_y, val_x, val_y, test_x, test_y = _dataset.load_data(test_user)

        train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 3)
        val_x   = val_x.reshape(val_x.shape[0], val_x.shape[1], -1, 3)
        test_x  = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 3)

        train_x = np.transpose(train_x, [0, 2, 1, 3])
        val_x = np.transpose(val_x, [0, 2, 1, 3])
        test_x  = np.transpose(test_x, [0, 2, 1, 3])

    train_x = torch.FloatTensor(train_x)
    train_y = torch.LongTensor(train_y)
    test_x  = torch.FloatTensor(test_x)
    test_y  = torch.LongTensor(test_y)  
    
    train_data = TensorDataset(train_x, train_y)
    test_data = TensorDataset(test_x, test_y)    

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    test_data_loader  = DataLoader(test_data, batch_size=batch_size)           
    
    return train_data_loader, test_data_loader


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data_ts2vec(name='opp_24_12', test_user=0, sensor_independent=False):
    if 'opp' in name:
        _dataset = data.Opportunity(name=name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)

        invalid_feature = np.arange(0, 36)
        train_x = np.delete(train_x, invalid_feature, axis=2)
        test_x = np.delete(test_x, invalid_feature, axis=2)

        train_x = train_x[:, :, :45]
        test_x = test_x[:, :, :45]

        if sensor_independent:
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 9)
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 9)
            train_x = np.transpose(train_x, [0, 2, 1, 3])
            test_x = np.transpose(test_x, [0, 2, 1, 3])

    elif 'realdisp' in name:
        _dataset = data.Realdisp(name=name)
        train_x, train_y, test_x, test_y = _dataset.load_data(test_user)

        if sensor_independent:
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 13)
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 13)
            train_x = np.transpose(train_x, [0, 2, 1, 3])
            test_x = np.transpose(test_x, [0, 2, 1, 3])

        train_x = train_x[:, :, :, :9]
        test_x = test_x[:, :, :, :9]

    elif 'skoda' in name:
        _dataset = data.Skoda(name=name)
        train_x, train_y, val_x, val_y, test_x, test_y = _dataset.load_data(test_user)

        if sensor_independent:
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1, 3)
            val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], -1, 3)
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], -1, 3)

            train_x = np.transpose(train_x, [0, 2, 1, 3])
            val_x = np.transpose(val_x, [0, 2, 1, 3])
            test_x = np.transpose(test_x, [0, 2, 1, 3])



    # train_data = TensorDataset(train_x, train_y)
    # test_data = TensorDataset(test_x, test_y)
    #
    # train_data_loader = DataLoader(train_data, batch_size=batch_size)
    # test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_x,train_y,test_x,test_y



def get_info_params(model):
    """
    Display a summary of trainable/frozen network parameter counts
    :param model:
    :return:
    """
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(paint(f"[-] {num_trainable}/{num_total} trainable parameters", "blue"))


def get_info_layers(model):
    """
    Display network layer information
    :param model:
    :return:
    """
    print("Layer Name \t\t Parameter Size")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t\t", model.state_dict()[param_tensor].size())
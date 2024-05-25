'''
Use the generated dataset for TLC and CC to construct surrogate model
'''
import scipy
from sklearn import linear_model
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import  os
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import mordred.CPSA as CPSA
import mordred
from mordred import Calculator, descriptors,is_missing
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from rdkit.Chem import MACCSkeys
from PIL import Image
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import warnings
import heapq
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pandas
from collections import deque
import numpy as np
import random
from utils import *

random.seed(525)
np.random.seed(1101)
torch.manual_seed(324)
batch_size=2048
def parse_args():
    parser = argparse.ArgumentParser()
    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

class ANN_TLC(nn.Module):
    '''
    Construct artificial neural network for surrogate model of TLC
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN_TLC, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)


    def forward(self, x):
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        x = F.sigmoid(x)
        return x

class ANN_CC(nn.Module):
    '''
        Construct artificial neural network for surrogate model of CC
        '''

    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN_CC, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)

    def forward(self, x):
        #x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0] + 1e-8)
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        return x

def split_dataset(data):
    '''
    split dataset for colum chromatography
    :param data: dataset
    :return: split dataset
    '''
    if 'X_CC.npy' not in os.listdir('dataset_save'):
        all_eluent_ratio = []
        all_descriptor=np.load('dataset_save/all_descriptor_CC.npy')
        dataset = []
        y=[]
        for i in tqdm(range(len(data['smiles']))):
            if data['t1'][i]*data['speed'][i]>60:
                continue
            if data['t2'][i] * data['speed'][i]>120:
                continue
            smile = data['smiles'][i]
            mol = Chem.MolFromSmiles(smile)
            Finger = MACCSkeys.GenMACCSKeys(mol)
            sub_data = [x for x in Finger]
            MolWt = Descriptors.ExactMolWt(mol)
            nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds

            HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
            HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
            LogP = Descriptors.MolLogP(mol)  # LogP
            sub_data.extend([MolWt, nRotB, HBD, HBA, LogP])
            #the added descriptor is selected by correlation matrix
            sub_data.extend(all_descriptor[i,[153,278,884,885,1273,1594,431,1768,1769,1288,1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i],data['m'][i],data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i]*data['speed'][i],data['t2'][i]*data['speed'][i]])
            all_eluent_ratio.append(data['eluent_ratio'][i])
        X=np.array(dataset)
        y=np.array(y)
        all_eluent_ratio=np.array(all_eluent_ratio)
        np.save('dataset_save/X_CC.npy', X)
        np.save('dataset_save/Y_CC.npy', y)
        np.save('dataset_save/eluent_ratio.npy', all_eluent_ratio)
    else:
        X=np.load('dataset_save/X_CC.npy')
        y=np.load('dataset_save/Y_CC.npy')
    np.save('dataset_save/X_max_CC.npy',np.max(X,axis=0))
    np.save('dataset_save/X_min_CC.npy', np.min(X, axis=0))
    X = (X - np.min(X,axis=0)) / (np.max(X,axis=0)- np.min(X,axis=0)+ 1e-8)
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    train_num = int(0.8 * X.shape[0])
    val_num = int(0.1 * X.shape[0])
    test_num = int(0.1 * X.shape[0])
    X_train= X[0:train_num]
    y_train = y[0:train_num]
    X_valid = X[train_num:train_num + val_num]
    y_valid=y[train_num:train_num + val_num]
    X_test = X[train_num + val_num:train_num + val_num + test_num]
    y_test= y[train_num + val_num:train_num + val_num + test_num]
    return X_train,y_train,X_valid,y_valid,X_test,y_test

def split_dataset_TLC(data):
    '''
      split dataset for TLC
      :param data: dataset
      :return: split dataset
      '''
    dataset = []
    y = []
    if 'X_TLC.npy' not in os.listdir('dataset_save'):
        all_descriptor=np.load('dataset_save/all_descriptor_TLC.npy')
        for i in tqdm(range(len(data['smiles']))):
            smile = data['smiles'][i]
            mol = Chem.MolFromSmiles(smile)
            Finger = MACCSkeys.GenMACCSKeys(mol)
            sub_data = [x for x in Finger]
            MolWt = Descriptors.ExactMolWt(mol)
            nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
            HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
            HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
            LogP = Descriptors.MolLogP(mol)  # LogP
            sub_data.extend([MolWt,  nRotB, HBD, HBA, LogP])
            sub_data.extend(all_descriptor[i, [153, 278, 884, 885, 1273, 1594, 431, 1768,1769, 1288, 1521]])
            eluent_array=np.array([data['H'][i],data['EA'][i],data['DCM'][i],data['MeOH'][i],data['Et2O'][i]])
            sub_data.extend(get_eluent_descriptor(eluent_array).tolist())
            dataset.append(sub_data)
            y.append([data['Rf'][i]])
        X = np.array(dataset)
        y = np.array(y)
        np.save('dataset_save/X_TLC.npy', X)
        np.save('dataset_save/Y_TLC.npy', y)
    else:
        X = np.load('dataset_save/X_TLC.npy')
        y = np.load('dataset_save/Y_TLC.npy')

    np.save('dataset_save/X_max_TLC.npy', np.max(X, axis=0))
    np.save('dataset_save/X_min_TLC.npy', np.min(X, axis=0))
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0) + 1e-8)
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
    train_num = int(0.8 * X.shape[0])
    val_num = int(0.1 * X.shape[0])
    test_num = int(0.1 * X.shape[0])
    X_train = X[0:train_num]
    y_train = y[0:train_num]
    X_valid = X[train_num:train_num + val_num]
    y_valid = y[train_num:train_num + val_num]
    X_test = X[train_num + val_num:train_num + val_num + test_num]
    y_test = y[train_num + val_num:train_num + val_num + test_num]
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def calaculate_metric(y_test,y_pred):
    MSE = np.sum(np.abs(y_test - y_pred) ** 2) / y_test.shape[0]
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
    R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    print(MSE,RMSE,MAE,R_square)

add_eluent_CC=True
config=parse_args()
Data_CC=read_data_CC()
Data_TLC=read_data_TLC()

# if first use, calculate the descriptors by mordred. If have calculsted, please denote it
calcu_mord_CC()
calcu_mord_TLC()

X_train_CC,y_train_CC,X_valid_CC,y_valid_CC,X_test_CC,y_test_CC=split_dataset(Data_CC)
X_train_TLC,y_train_TLC,X_valid_TLC,y_valid_TLC,X_test_TLC,y_test_TLC=split_dataset_TLC(Data_TLC)
Net_TLC=ANN_TLC(189,256,1).to(config.device)
if add_eluent_CC==True:
    Net_CC=ANN_CC(193,256,2).to(config.device)
    if add_Rf==False:
        Net_CC = ANN_CC(192, 256, 2).to(config.device)
else:
    Net_CC=ANN_CC(187,256,2).to(config.device)
X_train_TLC = Variable(torch.from_numpy(X_train_TLC.astype(np.float32)).to(config.device), requires_grad=True)
y_train_TLC= Variable(torch.from_numpy(y_train_TLC.astype(np.float32)).to(config.device))
X_valid_TLC = Variable(torch.from_numpy(X_valid_TLC.astype(np.float32)).to(config.device), requires_grad=True)
y_valid_TLC = Variable(torch.from_numpy(y_valid_TLC.astype(np.float32)).to(config.device))
X_test_TLC = Variable(torch.from_numpy(X_test_TLC.astype(np.float32)).to(config.device), requires_grad=True)
y_test_TLC = Variable(torch.from_numpy(y_test_TLC.astype(np.float32)).to(config.device))
print(X_train_TLC.shape,y_train_TLC.shape)
print(X_train_CC.shape,y_train_CC.shape)
model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
dir_name = 'model_save/'+model_name
optimizer_CC = torch.optim.Adam([{'params': Net_CC.parameters(), 'lr': config.NN_lr}])
optimizer_TLC = torch.optim.Adam([{'params': Net_TLC.parameters(), 'lr': config.NN_lr}])


if add_eluent_CC==False:
    X_train_CC=np.delete(X_train_CC,[183,184,185,186,187,188],axis=1)
    X_valid_CC=np.delete(X_valid_CC,[183,184,185,186,187,188],axis=1)
    X_test_CC=np.delete(X_test_CC,[183,184,185,186,187,188],axis=1)
X_train_CC = Variable(torch.from_numpy(X_train_CC.astype(np.float32)).to(config.device), requires_grad=True)
y_train_CC = Variable(torch.from_numpy(y_train_CC.astype(np.float32)).to(config.device))
X_valid_CC = Variable(torch.from_numpy(X_valid_CC.astype(np.float32)).to(config.device), requires_grad=True)
y_valid_CC = Variable(torch.from_numpy(y_valid_CC.astype(np.float32)).to(config.device))
X_test_CC = Variable(torch.from_numpy(X_test_CC.astype(np.float32)).to(config.device), requires_grad=True)
y_test_CC = Variable(torch.from_numpy(y_test_CC.astype(np.float32)).to(config.device))
mode='Test_CC'
n_epochs=10000

if mode=='Pre_Train_TLC':
    '''
    Train the surrogate model for TLC
    '''
    dir_name = 'model_save/' + 'model_TLC'
    try:
        os.makedirs(dir_name)
    except OSError:
        pass
    with open(dir_name + '/' + 'data.txt', 'a+') as f:
        for epoch in range(n_epochs):
            optimizer_TLC.zero_grad()
            prediction_TLC = Net_TLC(X_train_TLC)
            MSELoss_TLC = torch.nn.MSELoss()
            loss_TLC = MSELoss_TLC(y_train_TLC, prediction_TLC)
            loss_TLC.backward()
            optimizer_TLC.step()
            if (epoch + 1) % 100 == 0:
                pred_valid_TLC = Net_TLC(X_valid_TLC)
                valid_loss_TLC = MSELoss_TLC(y_valid_TLC, pred_valid_TLC)
                print(f"iter_num: {epoch+1}      loss_TLC: {loss_TLC.item()}   valid_TLC:{valid_loss_TLC.item()} ")
                f.write(f"iter_num: {epoch+1}      loss_TLC: {loss_TLC.item()}  valid_TLC:{valid_loss_TLC.item()}\n")
                torch.save(Net_TLC.state_dict(), dir_name + '/' + f'model_TLC_{epoch+1}.pkl')


if mode=='Pre_Train_CC':
    dir_name = 'model_save/'+'model_CC'
    try:
        os.makedirs(dir_name)
    except OSError:
        pass

    with open(dir_name + '/' + 'data.txt', 'a+') as f:  # 设置文件对象
        for epoch in tqdm(range(20000)):
            optimizer_CC.zero_grad()
            prediction_CC_1 = Net_CC(X_train_CC)[:,0].reshape(-1,1)
            prediction_CC_2 = Net_CC(X_train_CC)[:,1].reshape(-1,1)
            MSELoss_CC = torch.nn.MSELoss()
            loss_CC_1 = MSELoss_CC(y_train_CC[:, 0].reshape(-1, 1), prediction_CC_1)
            loss_CC_2 = MSELoss_CC(y_train_CC[:, 1].reshape(-1, 1), prediction_CC_2)
            loss =loss_CC_1 + loss_CC_2
            loss.backward()
            optimizer_CC.step()
            if (epoch + 1) % 100 == 0:
                pred_valid_CC = Net_CC(X_valid_CC)
                valid_loss_CC = MSELoss_CC(y_valid_CC, pred_valid_CC)
                print(
                    f"iter_num: {epoch + 1}     loss_CC_1:{loss_CC_1.item()}   loss_CC_2:{loss_CC_2.item()}  \n"
                    f" valid_CC:{valid_loss_CC.item()}  ")
                f.write(
                    f"iter_num: {epoch + 1}      loss_CC_1:{loss_CC_1.item()}   loss_CC_2:{loss_CC_2.item()}"
                    f"valid_CC:{valid_loss_CC.item()}  \n   ")
                torch.save(Net_CC.state_dict(), dir_name + '/' + f'model_{epoch + 1}.pkl')
                torch.save(optimizer_CC.state_dict(), dir_name + '/' + f'optimizer_{epoch + 1}.pkl')
            if (epoch + 1) % 2500 == 0:
                for p in optimizer_CC.param_groups:
                    p['lr'] *= 0.9
                print('adjust lr:', optimizer_CC.state_dict()['param_groups'][0]['lr'])


if mode=='Test_TLC':
    dir_name='model_save/model_TLC/'
    Net_TLC.load_state_dict(torch.load('model_save/model_TLC/model_TLC_1600.pkl'))
    Net_TLC.eval()
    pred_test_TLC=Net_TLC(X_test_TLC)
    calaculate_metric(pred_test_TLC.cpu().data.numpy().reshape(-1, ),
                      y_test_TLC.cpu().data.numpy().reshape(-1, ))

    df=pd.DataFrame({'pred_TLC':pred_test_TLC.cpu().data.numpy().reshape(-1, ),'true_TLC':y_test_TLC.cpu().data.numpy().reshape(-1, )})
    df.to_csv('result_save/pred_Rf.csv')
    plt.scatter(y_test_TLC.cpu().data.numpy().reshape(-1, ),pred_test_TLC.cpu().data.numpy().reshape(-1, ))
    plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01))
    plt.show()

if mode=='Test_CC':
    Net_CC.load_state_dict(torch.load('model_save/model_CC/model_2300.pkl'))
    pred_test_CC=Net_CC(X_test_CC)
    calaculate_metric(pred_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                      y_test_CC[:, 0].cpu().data.numpy().reshape(-1, ))
    calaculate_metric(pred_test_CC[:, 1].cpu().data.numpy().reshape(-1, ),
                      y_test_CC[:, 1].cpu().data.numpy().reshape(-1, ))
    df = pd.DataFrame({'pred_t1': pred_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                       'true_t1': y_test_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                       'pred_t2': pred_test_CC[:, 1].cpu().data.numpy().reshape(-1, ),
                       'true_t2': y_test_CC[:, 1].cpu().data.numpy().reshape(-1, ),
                       })
    df.to_csv('result_save/pred_CC.csv')
    plt.scatter(y_test_CC[:,0].cpu().data.numpy().reshape(-1, ),pred_test_CC[:,0].cpu().data.numpy().reshape(-1, ))
    plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01))
    plt.show()
    print('pred_test_CC:', pred_test_CC[0:5])
    print('pred_true_CC:', y_test_CC[0:5])

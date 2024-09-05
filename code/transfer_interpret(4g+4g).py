import  os
import scipy
import pandas as pd
from sklearn import linear_model
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import pymysql
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import mordred.CPSA as CPSA
import mordred
from multiprocessing import Process,Queue
from mordred import Calculator, descriptors,is_missing
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from rdkit.Chem import MACCSkeys
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pandas
import numpy as np
import random
from utils import *
import warnings
import copy
import pickle
from matplotlib.ticker import MaxNLocator
import sklearn
from sklearn.linear_model import RANSACRegressor
import joblib

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
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron):
        super(ANN_TLC, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)


    def forward(self, x):
        #x=(x-x.min(dim=0)[0])/(x.max(dim=0)[0]-x.min(dim=0)[0]+1e-8)
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
        #x=torch.clamp(x,min=0,max=1)
        return x

class ANN_CC(nn.Module):
    '''
    Construct artificial neural network
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

def split_dataset_transfer(data):
    if 'X_CC_8.npy' not in os.listdir('dataset_save'):
        all_eluent_ratio = []
        all_descriptor=np.load('dataset_save/all_descriptor_CC_8.npy')
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
            sub_data.extend(all_descriptor[i,[153,278,884,885,1273,1594,431,1768,1769,1288,1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i],data['m'][i],data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i]*data['speed'][i],data['t2'][i]*data['speed'][i]])
            all_eluent_ratio.append(data['eluent_ratio'][i])
        X=np.array(dataset)
        y=np.array(y)
        np.save('dataset_save/X_CC_8.npy', X)
        np.save('dataset_save/Y_CC_8.npy', y)
        np.save('dataset_save/eluent_ratio_8.npy', all_eluent_ratio)
    else:
        X=np.load('dataset_save/X_CC_8.npy')
        y=np.load('dataset_save/Y_CC_8.npy')
    np.save('dataset_save/X_max_CC_8.npy',np.max(X,axis=0))
    np.save('dataset_save/X_min_CC_8.npy', np.min(X, axis=0))
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
        #X = np.hstack((X, np.zeros([X.shape[0], 3])))
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

def calaculate_metric(y_test, y_pred,delete_outliers=False):
    #abs_e=np.abs(y_test-y_pred)/y_pred
    if delete_outliers==True:
        delta = np.abs((y_test - y_pred)).reshape(-1, ).tolist()
        outliers = list(map(delta.index, heapq.nlargest(int(y_test.shape[0] * 0.05), delta)))
        y_test = np.delete(y_test, outliers, axis=0)
        y_pred = np.delete(y_pred, outliers, axis=0)

    MSE = np.sum(np.abs(y_test - y_pred) ** 2) / y_test.shape[0]
    RMSE = np.sqrt(MSE)
    MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
    R_square = 1 - (((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum())
    print(MSE, RMSE, MAE, R_square)

def plot_result(y_input, y_pred):
    calaculate_metric(y_input, y_pred)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 1, figsize=(2, 2), dpi=300)
    plt.scatter(y_input, y_pred, c='#8A83B4', s=7, alpha=0.25)
    plt.plot(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), linewidth=1.5, linestyle='--', color='black')
    plt.yticks(fontproperties='Arial', size=6)
    plt.xticks(fontproperties='Arial', size=6)
    axes.yaxis.set_major_locator(MaxNLocator(5))
    axes.xaxis.set_major_locator(MaxNLocator(5))
    plt.savefig(f'plot_save/regression_plot/all.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'plot_save/regression_plot/all.png', bbox_inches='tight', dpi=300)
    plt.show()

eluent_array = np.array([[86.10955045,  0.,          3.,          0.,          0.,          2.5866    ],  #1/0
                [8.61287869e+01, 2.60396040e-01, 2.98019802e+00, 0.00000000e+00, 1.98019802e-02,2.56662772e+00], #100/1
                [8.61476461e+01, 5.15686275e-01, 2.96078431e+00, 0.00000000e+00, 3.92156863e-02, 2.54704706e+00],  ##50/1
                #[8.61569377e+01, 6.41463415e-01, 2.95121951e+00, 0.00000000e+00,4.87804878e-02, 2.53740000e+00], ##40/1
                #[8.61722240e+01, 8.48387097e-01, 2.93548387e+00, 0.00000000e+00, 6.45161290e-02, 2.52152903e+00], ##30/1
                [86.2020685,   1.25238095,  2.9047619,   0. ,         0.0952381,   2.49054286],  ##20/1
                [86.28617582,  2.39090909,  2.81818182,  0. ,         0.18181818,  2.40321818],  ##10/1
                [86.43336362,  4.38333333,  2.66666667,  0.,          0.33333333,  2.2504    ],  ##5/1
                [86.7571768,   8.76666667,  2.33333333,  0.,          0.66666667,  1.9142    ],  ##2/1
                [87.08098997, 13.15,        2.,          0. ,         1.,          1.578 ],  ##1/1
                [88.0524295, 26.3, 1., 0., 2., 0.5694]])  ##0/1
eluent_ratio=np.array([1,100/101,50/51,20/21,10/11,5/6,2/3,1/2,0])

config=parse_args()
Data_CC=read_data_CC_8()
Data_TLC=read_data_TLC()
calcu_mord_CC_8()
X_train_CC, y_train_CC, X_valid_CC, y_valid_CC, X_test_CC, y_test_CC = split_dataset_transfer(Data_CC)
X_train_TLC, y_train_TLC, X_valid_TLC, y_valid_TLC, X_test_TLC, y_test_TLC = split_dataset_TLC(Data_TLC)
Net_TLC=ANN_TLC(189,256,1).to(config.device)
Net_CC=ANN_CC(192,256,2).to(config.device)
Net_TLC.load_state_dict(torch.load('model_save/model_TLC/model_TLC_1600.pkl'))
Net_CC.load_state_dict(torch.load('model_save/model_CC/model_2300.pkl'))
X_CC = np.load('dataset_save/X_CC_8.npy')
y_CC= np.load('dataset_save/Y_CC_8.npy')
X_CC_unique = np.load('dataset_save/X_CC_unique.npy')
y_CC_unique= np.load('dataset_save/Y_CC_unique.npy')
X_CC_unique_compound = np.load('dataset_save/X_CC_unique_compound.npy')
y_CC_unique_compound= np.load('dataset_save/Y_CC_unique_compound.npy')
X_CC_max=np.load('dataset_save/X_max_CC_8.npy')
X_CC_min=np.load('dataset_save/X_min_CC_8.npy')
X_CC=(X_CC-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
X_CC_unique=(X_CC_unique-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
X_CC_unique_compound=(X_CC_unique_compound-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
eluent_array=(eluent_array-X_CC_min[183:189])/(X_CC_max[183:189]-X_CC_min[183:189]+ 1e-8)


def test_all_eq_origin(X_CC,y_CC):
    Rf = Net_TLC(
        Variable(torch.from_numpy(X_CC[:, 0:189].astype(np.float32)).to(config.device), requires_grad=True))
    X_CC = Variable(torch.from_numpy(X_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_CC = Variable(torch.from_numpy(y_CC.astype(np.float32)).to(config.device))
    df = pd.DataFrame({'Rf': Rf.cpu().data.numpy().reshape(-1, ),
                       't1': y_CC[:, 0].cpu().data.numpy().reshape(-1, ),
                       't2': y_CC[:, 1].cpu().data.numpy().reshape(-1, )})
    X_CC_origin = np.load('dataset_save/X_CC_8.npy')
    eluent_ratio = np.load('dataset_save/eluent_ratio_8.npy').reshape(-1, 1)
    X_CC_origin = np.hstack((X_CC_origin, eluent_ratio))
    pred_t1 = eluent_ratio / (0.1472 * Rf.cpu().data.numpy().reshape(-1, 1) + 0.0114)
    pred_t2 = eluent_ratio / (0.0689 * Rf.cpu().data.numpy().reshape(-1, 1) + 0.0054)
    pred_t1[pred_t1 == 0] = 5.147
    pred_t2[pred_t2 == 0] = 10.980
    df_pred = pd.DataFrame({'pred_t1': pred_t1.reshape(-1, ), 'pred_t2': pred_t2.reshape(-1, )})
    df_X = pd.DataFrame(X_CC_origin)
    df = pd.concat([df_X, df, df_pred], axis=1)
    split_array = np.arange(0,1.1,0.1)
    all_dt=[]
    for i in tqdm(range(pred_t1.shape[0])):
        for index in range(split_array.shape[0]-1):
            if split_array[index]<df['Rf'].values[i]<split_array[index+1]:
                all_X=df.values

                if index==0:
                    model = DecisionTreeRegressor(criterion='mae', max_depth=9)
                    model=joblib.load(f'model_save_tree/tree_{index}_9_both.pkl')
                if index != 0:
                    model = DecisionTreeRegressor(criterion='mae', max_depth=5)
                    model = joblib.load(f'model_save_tree/tree_{index}_9_both.pkl')
                dt=model.predict(all_X[i,0:183].reshape(1, -1))
                pred_t1[i]*=dt
                pred_t2[i] *= dt
                all_dt.append(dt)

    all_dt = np.array(all_dt)
    df['pred_t1'] = pred_t1
    df['pred_t2'] = pred_t2
    b = df['pred_t1'].values.reshape(-1, 1)
    a = df['t1'].values.reshape(-1, 1)
    b_2 = df['pred_t2'].values.reshape(-1, 1)
    a_2 = df['t2'].values.reshape(-1, 1)
    save_result = pd.DataFrame({'t1': a.reshape(-1, ), 'pred_t1': b.reshape(-1, ),
                                't2': a_2.reshape(-1, ), 'pred_t2': b_2.reshape(-1, ),
                                'Rf': Rf.cpu().data.numpy().reshape(-1, 1).reshape(-1, ),
                                'eluent_ratio': eluent_ratio.reshape(-1, ),
                                'ratio': (pred_t1 / all_dt).reshape(-1, ),
                                'epi': all_dt.reshape(-1, ),
                                't1/epi': (a / all_dt).reshape(-1, ),
                                't2/epi': (a_2 / all_dt).reshape(-1, )
                                })
    print(a.shape, b.shape)
    print(1 - (((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()))
    print(1 - (((a_2 - b_2) ** 2).sum() / ((a_2 - a_2.mean()) ** 2).sum()))
    save_result.to_csv(f"result_save/tree_adopt/result_8_origin.csv")
    with open(f"result_save/tree_adopt/R_2_8_origin.log", "w") as f:
        f.write(f'R_2: {1 - (((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum())}')
        f.write(f'R_2: {1 - (((a_2 - b_2) ** 2).sum() / ((a_2 - a_2.mean()) ** 2).sum())}')
    plt.plot(np.arange(0, 60, 1), np.arange(0, 60, 1))
    plt.scatter(a, b)
    plt.show()
    plt.plot(np.arange(0, 120, 1), np.arange(0, 120, 1))
    plt.scatter(a_2, b_2)
    plt.show()
    
def find_transfer_param():
    def model(x, a, b, c):
        return a * x[0] / (b * x[1] + c)

    from scipy.optimize import curve_fit
    result = pd.read_csv('result_save/tree_adopt/result_8_origin.csv')
    result = result[result['t1/epi'] < 1000]
    result = result[result['t2/epi'] < 1000]
    t1 = result['t1'].values
    pred_t1 = result['pred_t1'].values
    t2 = result['t2'].values
    pred_t2 = result['pred_t2'].values
    eluent_ratio = result['eluent_ratio'].values
    epi = result['epi'].values
    Rf = result['Rf'].values
    df = result
    split_array = np.arange(0, 1.1, 0.1)
    x = np.arange(0.05, 1.05, 0.1)
    Rf_t1 = []
    Rf_t2 = []

    params, covariance = curve_fit(model, (df['eluent_ratio'], df['Rf']), df['t1/epi'])
    print(params)
    params, covariance = curve_fit(model, (df['eluent_ratio'], df['Rf']), df['t2/epi'])
    print(params)




# before transfer
test_all_eq_origin(X_CC,y_CC)
# find transfer param
find_transfer_param()





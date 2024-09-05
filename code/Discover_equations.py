import  os
import scipy
import pandas as pd
from sklearn import linear_model
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
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
from multiprocessing import Process,Queue
import julia
from mordred import Calculator, descriptors,is_missing
from sklearn.metrics import mean_squared_error
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from rdkit.Chem import MACCSkeys
from PIL import Image
import torch.nn as nn
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
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
import time
import warnings
import copy
import pickle
from matplotlib.ticker import MaxNLocator
import sklearn
from pysr import PySRRegressor
import pysr
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from IPython.display import Image, display
import pydotplus
import joblib

#use Graphviz to visualize the tree
os.environ["PATH"] += r";D:\software\Julia-1.9.3\bin"
os.environ["PATH"] += r';D:\software\Graphviz\bin'

#pysr.install()
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

def split_dataset(data):
    if 'X_CC.npy' not in os.listdir('dataset_save'):
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
            sub_data.extend(all_descriptor[i,[153,278,884,885,1273,1594,431,1768,1769,1288,1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i],data['m'][i],data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i]*data['speed'][i],data['t2'][i]*data['speed'][i]])
        X=np.array(dataset)
        y=np.array(y)
        np.save('dataset_save/X_CC.npy', X)
        np.save('dataset_save/Y_CC.npy', y)
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
    return X_train,y_train,X_valid,y_valid,X_test,y_test,state

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

def get_unqiue_compound(data):
    all_descriptor = np.load('dataset_save/all_descriptor_CC.npy')
    dataset = []
    y = []
    all_smile=[]
    for i in tqdm(range(len(data['smiles']))):
        if data['t1'][i] * data['speed'][i] > 60:
            continue
        if data['t2'][i] * data['speed'][i] > 120:
            continue
        smile = data['smiles'][i]
        if smile not in all_smile:
            mol = Chem.MolFromSmiles(smile)
            Finger = MACCSkeys.GenMACCSKeys(mol)
            sub_data = [x for x in Finger]
            MolWt = Descriptors.ExactMolWt(mol)
            nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
            HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
            HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
            LogP = Descriptors.MolLogP(mol)  # LogP
            sub_data.extend([MolWt, nRotB, HBD, HBA, LogP])
            sub_data.extend(all_descriptor[i, [153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]])
            sub_data.extend(data['eluent'][i])
            sub_data.extend([data['e'][i], data['m'][i], data['V_e'][i]])
            dataset.append(sub_data)
            y.append([data['t1'][i] * data['speed'][i], data['t2'][i] * data['speed'][i]])
            all_smile.append(smile)
        else:
            continue
    X = np.array(dataset)
    y = np.array(y)
    print(X.shape)
    np.save('dataset_save/X_CC_unique_compound.npy', X)
    np.save('dataset_save/Y_CC_unique_compound.npy', y)

def get_unqiue_eluent(data):
    all_descriptor = np.load('dataset_save/all_descriptor_CC.npy')
    dataset = []
    y = []
    all_data=[]
    all_smile=[]
    for i in tqdm(range(len(data['smiles']))):
        if data['t1'][i] * data['speed'][i] > 60:
            continue
        if data['t2'][i] * data['speed'][i] > 120:
            continue
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        Finger = MACCSkeys.GenMACCSKeys(mol)
        sub_data = [x for x in Finger]
        compare_data= [x for x in Finger]
        MolWt = Descriptors.ExactMolWt(mol)
        nRotB = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
        HBD = Descriptors.NumHDonors(mol)  # Number of H bond donors
        HBA = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
        LogP = Descriptors.MolLogP(mol)  # LogP
        sub_data.extend([MolWt, nRotB, HBD, HBA, LogP])
        sub_data.extend(all_descriptor[i, [153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]])
        compare_data.extend([MolWt, nRotB, HBD, HBA, LogP])
        compare_data.extend(all_descriptor[i, [153, 278, 884, 885, 1273, 1594, 431, 1768, 1769, 1288, 1521]])
        sub_data.extend(data['eluent'][i])
        sub_data.extend([data['e'][i], data['m'][i], data['V_e'][i]])
        compare_data.extend([data['e'][i], data['m'][i], data['V_e'][i]])
        if compare_data not in all_data:
            dataset.append(sub_data)
            y.append([data['t1'][i] * data['speed'][i], data['t2'][i] * data['speed'][i]])
            all_data.append(compare_data)
            all_smile.append(smile)
        else:
            continue
    X = np.array(dataset)
    y = np.array(y)
    print(X.shape)
    np.save('dataset_save/smile_unique.npy',all_smile,allow_pickle=True)
    np.save('dataset_save/X_CC_unique.npy', X)
    np.save('dataset_save/Y_CC_unique.npy', y)



def calaculate_metric(y_test, y_pred,delete_outliers=False):
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

# the descriptors of eluent array under different ratio
eluent_array = np.array([[86.10955045,  0.,          3.,          0.,          0.,          2.5866    ],  #1/0
                [8.61287869e+01, 2.60396040e-01, 2.98019802e+00, 0.00000000e+00, 1.98019802e-02,2.56662772e+00], #100/1
                [8.61476461e+01, 5.15686275e-01, 2.96078431e+00, 0.00000000e+00, 3.92156863e-02, 2.54704706e+00],  ##50/1
                [86.2020685,   1.25238095,  2.9047619,   0. ,         0.0952381,   2.49054286],  ##20/1
                [86.28617582,  2.39090909,  2.81818182,  0. ,         0.18181818,  2.40321818],  ##10/1
                [86.43336362,  4.38333333,  2.66666667,  0.,          0.33333333,  2.2504    ],  ##5/1
                [86.7571768,   8.76666667,  2.33333333,  0.,          0.66666667,  1.9142    ],  ##2/1
                [87.08098997, 13.15,        2.,          0. ,         1.,          1.578 ],  ##1/1
                [88.0524295, 26.3, 1., 0., 2., 0.5694]])
eluent_ratio=np.array([1,100/101,50/51,20/21,10/11,5/6,2/3,1/2,0])

config=parse_args()
Data_CC=read_data_CC()
Data_TLC=read_data_TLC()

# get the unique compound in dataset to generate meta-data
get_unqiue_compound(Data_CC)
get_unqiue_eluent(Data_CC)

X_train_CC, y_train_CC, X_valid_CC, y_valid_CC, X_test_CC, y_test_CC,state = split_dataset(Data_CC)
X_train_TLC, y_train_TLC, X_valid_TLC, y_valid_TLC, X_test_TLC, y_test_TLC = split_dataset_TLC(Data_TLC)
Net_TLC=ANN_TLC(189,256,1).to(config.device)
Net_CC=ANN_CC(192,256,2).to(config.device)
Net_TLC.load_state_dict(torch.load('model_save/model_TLC/model_TLC_1600.pkl'))
Net_CC.load_state_dict(torch.load('model_save/model_CC/model_2300.pkl'))
X_CC = np.load('dataset_save/X_CC.npy')
y_CC= np.load('dataset_save/Y_CC.npy')
X_CC_unique = np.load('dataset_save/X_CC_unique.npy')
y_CC_unique= np.load('dataset_save/Y_CC_unique.npy')
X_CC_unique_compound = np.load('dataset_save/X_CC_unique_compound.npy')
y_CC_unique_compound= np.load('dataset_save/Y_CC_unique_compound.npy')
X_CC_max=np.load('dataset_save/X_max_CC.npy')
X_CC_min=np.load('dataset_save/X_min_CC.npy')
X_CC=(X_CC-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
X_CC_unique=(X_CC_unique-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
X_CC_unique_compound=(X_CC_unique_compound-X_CC_min)/(X_CC_max-X_CC_min+1e-8)
eluent_array=(eluent_array-X_CC_min[183:189])/(X_CC_max[183:189]-X_CC_min[183:189]+ 1e-8)
print(X_CC_unique.shape)
print(X_CC_unique_compound.shape)

max_depth = 9

def save_Rf_t_per_eluent():
    '''
    generate meta-data and save coressponding V1(notation here is t1), V2(notation here is t2), and Rf
    '''
    for i in tqdm(range(X_CC_unique.shape[0])):
        X_input=np.zeros([eluent_array.shape[0],X_CC_unique.shape[1]])
        for j in range(eluent_array.shape[0]):
            X_input[j]=X_CC_unique[i]
            X_input[j,183:189]=eluent_array[j]
        Rf = Net_TLC(
            Variable(torch.from_numpy(X_input[:, 0:189].astype(np.float32)).to(config.device), requires_grad=True))
        pred_CC=Net_CC(Variable(torch.from_numpy(X_input.astype(np.float32)).to(config.device), requires_grad=True))
        all_Rf.append(Rf.cpu().data.numpy().reshape(-1,1))
        all_t_1.append(pred_CC[:,0].cpu().data.numpy().reshape(-1, 1))
        all_t_2.append(pred_CC[:,1].cpu().data.numpy().reshape(-1, 1))
    all_Rf=np.hstack(all_Rf).T
    all_t_1=np.hstack(all_t_1).T
    all_t_2=np.hstack(all_t_2).T
    np.save('result_save/all_Rf.npy',all_Rf)
    np.save('result_save/all_t_1.npy',all_t_1)
    np.save('result_save/all_t_2.npy',all_t_2)

def save_Rf_t():
    '''
    get the distribution of V1, V2 under different eluent ratios.
    Only situations larger than 9 samples are recorded.
    '''
    all_Rf=np.load('result_save/all_Rf.npy')
    all_t1=np.load('result_save/all_t_1.npy')
    all_t2=np.load('result_save/all_t_2.npy')
    split_array=np.arange(0,1.1,0.1)
    x=np.arange(0.05,1.05,0.1)
    Rf_t1=[]
    Rf_t2=[]
    for i in range(all_Rf.shape[1]):
        Rf=all_Rf[:,i]
        t1=all_t1[:,i]
        t2=all_t2[:,i]
        df=pd.DataFrame({'Rf':Rf,'t1':t1,'t2':t2})
        for index in range(len(split_array)-1):
            all_box_t1=(df[(df['Rf']>=split_array[index])&(df['Rf']<split_array[index+1])].values[:,1])
            all_box_t2=(df[(df['Rf'] >= split_array[index]) & (df['Rf'] < split_array[index + 1])].values[:, 2])
            if len(all_box_t1)>=9:
                all_box_t1=np.array(all_box_t1).reshape(-1,)
                Rf_t1.append([x[index],eluent_ratio[i], np.mean(all_box_t1), np.std(all_box_t1),
                              np.percentile(all_box_t1,10,axis=0),
                              np.percentile(all_box_t1,50,axis=0),
                              np.percentile(all_box_t1,90,axis=0)])
            if len(all_box_t2)>=9:
                all_box_t2 = np.array(all_box_t2).reshape(-1,)
                Rf_t2.append([x[index],eluent_ratio[i],all_box_t2.mean(),all_box_t2.std(),
                              np.percentile(all_box_t2, 10, axis=0),
                              np.percentile(all_box_t2, 50, axis=0),
                              np.percentile(all_box_t2, 90, axis=0)
                              ])
    Rf_t1=np.vstack(Rf_t1)
    Rf_t2=np.vstack(Rf_t2)
    np.save('result_save/Rf_t1.npy',Rf_t1)
    np.save('result_save/Rf_t2.npy',Rf_t2)

def find_eq_Rf_t_mean():
    Rf_t1=np.load('result_save/Rf_t1.npy')
    Rf_t2 =np.load('result_save/Rf_t2.npy')

    model = PySRRegressor(
        model_selection="best",  # Result is mix of simplicity+accuracy
        niterations=100,
        binary_operators=["+", "*"],
        unary_operators=[
            # "cos",
            # "exp",
            # "sin",
            "inv(x) = 1/x",
        # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        loss="loss(x, y) = (x - y)^2",
        # ^ Custom loss function (julia syntax)
    )
    model.fit(Rf_t1[:,0:2], Rf_t1[:,3])
    model.fit(Rf_t2[:,0:2], Rf_t2[:,3])

def save_result(X_CC,y_CC):
    Rf = Net_TLC(
        Variable(torch.from_numpy(X_CC[:, 0:189].astype(np.float32)).to(config.device), requires_grad=True))
    X_CC = Variable(torch.from_numpy(X_CC.astype(np.float32)).to(config.device), requires_grad=True)
    y_CC = Variable(torch.from_numpy(y_CC.astype(np.float32)).to(config.device))
    df=pd.DataFrame({'Rf':Rf.cpu().data.numpy().reshape(-1,),
                     't1':y_CC[:,0].cpu().data.numpy().reshape(-1,),
                     't2':y_CC[:,1].cpu().data.numpy().reshape(-1,),
                     'NN_pred_t1':Net_CC(X_CC)[:,0].cpu().data.numpy().reshape(-1,),
                     'NN_pred_t2':Net_CC(X_CC)[:,1].cpu().data.numpy().reshape(-1,)})
    X_CC_origin = np.load('dataset_save/X_CC.npy')
    eluent_ratio=np.load('dataset_save/eluent_ratio.npy').reshape(-1,1)
    X_CC_origin=np.hstack((X_CC_origin,eluent_ratio))

    # calculate the prediction through the discovered equations
    pred_t1=eluent_ratio/(0.1472*Rf.cpu().data.numpy().reshape(-1,1)+0.0114)
    pred_t2=eluent_ratio/(0.0689*Rf.cpu().data.numpy().reshape(-1,1)+0.0054)
    pred_t1[pred_t1==0]=5.147
    pred_t2[pred_t2==0]=10.980
    df_pred=pd.DataFrame({'pred_t1':pred_t1.reshape(-1,),'pred_t2':pred_t2.reshape(-1,)})
    df_X=pd.DataFrame(X_CC_origin)
    df=pd.concat([df_X,df,df_pred],axis=1)
    df.to_csv('result_save/Rf_to_CC.csv')
    plt.scatter(Rf.cpu().data.numpy(),y_CC[:,0].cpu().data.numpy())
    plt.show()
    plt.scatter(Rf.cpu().data.numpy(), y_CC[:, 1].cpu().data.numpy())
    plt.show()
    plt.scatter(Rf.cpu().data.numpy(), (y_CC[:, 1]-y_CC[:,0]).cpu().data.numpy())
    plt.show()



def find_all_eq_tree_both():
    all_Rf = np.load('result_save/all_Rf.npy')
    all_t1 = np.load('result_save/all_t_1.npy')
    all_t2 = np.load('result_save/all_t_2.npy')
    X_CC_unique = np.load('dataset_save/X_CC_unique.npy')
    y_CC_unique = np.load('dataset_save/Y_CC_unique.npy')
    split_array = np.arange(0,1.1,0.1)
    x = np.arange(0.05, 1.05, 0.1)
    for index in range(len(split_array)-1):
        all_X = []
        Rf_t1 = []
        Rf_t2 = []
        for i in range(all_Rf.shape[1]):
            Rf = all_Rf[:, i]
            t1 = all_t1[:, i]
            t2 = all_t2[:, i]
            pred_t1=eluent_ratio[i]/(0.1472*Rf.reshape(-1,)+0.0114)
            pred_t2=eluent_ratio[i]/(0.0689*Rf.reshape(-1,)+0.0054)
            pred_t1[pred_t1==0]=5.147
            pred_t2[pred_t2==0]=10.980
            df = pd.DataFrame({'Rf': Rf, 't1': t1, 't2': t2,'pred_t1':pred_t1,'pred_t2':pred_t2})
            df_X=pd.DataFrame(X_CC_unique)
            df=pd.concat([df_X,df],axis=1)
            df_t1 = df[(df['Rf'] >= split_array[index]) & (df['Rf'] < split_array[index+1])]
            df_t1 = df_t1[(df_t1['t1'] <= 60) & (df_t1['t2'] <= 120)]
            all_X.append(df_t1.values)
        all_X=np.vstack(all_X)
        X=np.vstack((all_X[:,0:183],all_X[:,0:183]))
        y=np.vstack(((all_X[:,193]/all_X[:,195]).reshape(-1,1),(all_X[:,194]/all_X[:,196]).reshape(-1,1)))
  
        model = DecisionTreeRegressor(criterion='mae', max_depth=max_depth)
   
        model.fit(X,y)

        score = model.score(X, y)
        print("Score：", score)
        text_representation = tree.export_text(model)
        joblib.dump(model,f'model_save_tree/tree_{index}_{max_depth}_both.pkl')
        # Save rules
        try:
            os.makedirs(f'result_save/tree_depth_{max_depth}/')
        except OSError:
            pass

        with open(f"result_save/tree_depth_{max_depth}/decistion_tree_{index}_both.log", "w") as f:
            f.write(f'Score: {score}\n')
            f.write(text_representation)

def draw_the_tree():
    all_Rf = np.load('result_save/all_Rf.npy')
    all_t1 = np.load('result_save/all_t_1.npy')
    all_t2 = np.load('result_save/all_t_2.npy')
    X_CC_unique = np.load('dataset_save/X_CC_unique.npy')
    y_CC_unique = np.load('dataset_save/Y_CC_unique.npy')
    split_array = np.arange(0, 1.1, 0.1)
    x = np.arange(0.05, 1.05, 0.1)
    for index in range(len(split_array) - 1):
        all_X = []
        Rf_t1 = []
        Rf_t2 = []
        for i in range(all_Rf.shape[1]):
            Rf = all_Rf[:, i]
            t1 = all_t1[:, i]
            t2 = all_t2[:, i]
            pred_t1 = eluent_ratio[i] / (0.1472 * Rf.reshape(-1, ) + 0.0114)
            pred_t2 = eluent_ratio[i] / (0.0689 * Rf.reshape(-1, ) + 0.0054)
            pred_t1[pred_t1 == 0] = 5.147
            pred_t2[pred_t2 == 0] = 10.980
            df = pd.DataFrame({'Rf': Rf, 't1': t1, 't2': t2, 'pred_t1': pred_t1, 'pred_t2': pred_t2})
            df_X = pd.DataFrame(X_CC_unique)
            df = pd.concat([df_X, df], axis=1)
            df_t1 = df[(df['Rf'] >= split_array[index]) & (df['Rf'] < split_array[index + 1])]
            df_t1 = df_t1[(df_t1['t1'] <= 60) & (df_t1['t2'] <= 120)]
            all_X.append(df_t1.values)
        all_X = np.vstack(all_X)
        X = np.vstack((all_X[:, 0:183], all_X[:, 0:183]))
        y = np.vstack(
            ((all_X[:, 193] / all_X[:, 195]).reshape(-1, 1), (all_X[:, 194] / all_X[:, 196]).reshape(-1, 1)))
        for i in range(y.shape[0]):
            if y[i]<=0.5:
                y[i]=1
            elif (0.5<y[i]) and (y[i]<= 1):
                y[i]=2
            else:
                y[i]=3


        model = tree.DecisionTreeClassifier(max_depth=9)
        model.fit(X, y)
        score = model.score(X, y)
        print("Score：", score)
        text_representation = tree.export_text(model)
        joblib.dump(model, f'model_save_classifed_tree/tree_{index}_{max_depth}_both.pkl')
        try:
            os.makedirs(f'result_save_classified_tree/tree_depth_{max_depth}/')
        except OSError:
            pass

        with open(f"result_save_classified_tree/tree_depth_{max_depth}/decistion_tree_{index}_both.log", "w") as f:
            f.write(f'Score: {score}\n')
            f.write(text_representation)



save_Rf_t_per_eluent()
save_Rf_t()
find_eq_Rf_t_mean()
save_result(X_CC,y_CC)
find_all_eq_tree_both()
draw_the_tree()

import numpy as np
import pandas as pd
from tqdm import  tqdm
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors, is_missing
Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']
def get_descriptor(smiles,ratio):
    compound_mol = Chem.MolFromSmiles(smiles)
    descriptor=[]
    descriptor.append(Descriptors.ExactMolWt(compound_mol))
    descriptor.append(Chem.rdMolDescriptors.CalcTPSA(compound_mol))
    descriptor.append(Descriptors.NumRotatableBonds(compound_mol))  # Number of rotable bonds
    descriptor.append(Descriptors.NumHDonors(compound_mol))  # Number of H bond donors
    descriptor.append(Descriptors.NumHAcceptors(compound_mol)) # Number of H bond acceptors
    descriptor.append(Descriptors.MolLogP(compound_mol)) # LogP
    descriptor=np.array(descriptor)*ratio
    return descriptor

def get_eluent_descriptor(eluent_array):
    eluent=eluent_array
    des = np.zeros([6,])
    for i in range(eluent.shape[0]):
        if eluent[i] != 0:
            e_descriptors = get_descriptor(Eluent_smiles[i], eluent[i])
            des+=e_descriptors
    return des

def obtain_3D_mol(smiles,name):
    mol = AllChem.MolFromSmiles(smiles)
    new_mol = Chem.AddHs(mol)
    res = AllChem.EmbedMultipleConfs(new_mol)
    ### MMFF generates multiple conformations
    res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
    new_mol = Chem.RemoveHs(new_mol)
    Chem.MolToMolFile(new_mol, name+'.mol')
    return new_mol
def convert_eluent(eluent):
    ratio=[]
    for e in eluent:
        PE=int(e.split('/')[0])
        EA=int(e.split('/')[1])
        ratio.append(get_eluent_descriptor(np.array([PE,EA,0,0,0])/(PE+EA)))
    return np.vstack(ratio)
def convert_e(e):
    new_e=np.zeros([e.shape[0],])
    for i in range(len(e)):
        if e[i]=='PE':
            new_e[i]=0
        elif e[i]=='EA':
            new_e[i]=1
        elif e[i]=='DCM':
            new_e[i]=2
        else:
            print(e)
    return new_e

def convert_eluent_ratio(eluent):
    ratio = []
    for e in eluent:
        PE = int(e.split('/')[0])
        EA = int(e.split('/')[1])
        ratio.append(PE / (PE + EA))
    return np.vstack(ratio)

def read_data_CC():
    df=pd.read_excel('dataset_1013.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 4g']
    df=df[np.isnan(df.t1)==False]
    t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['density g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['loading volume/ul'].values
    e=df['loading solvent'].values
    eluent=df['PE/EA'].values
    speed=df['flow rate ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,'eluent_ratio':eluent_ratio}
    return data

def read_data_CC_8():
    df=pd.read_excel('dataset_4+4g_1020.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 4g+4g']
    df=df[np.isnan(df.t1)==False]
   t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['density g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['loading volume/ul'].values
    e=df['loading solvent'].values
    eluent=df['PE/EA'].values
    speed=df['flow rate ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,'eluent_ratio':eluent_ratio}
    return data
def read_data_CC_25():
    #df=pd.read_excel('dataset_25g_1022.xlsx')
    df = pd.read_excel('dataset_25g_1218.xlsx')
    df=df[df.t1!=-1]
    df=df[df.column_specs=='Silica-CS 25g']
    df=df[np.isnan(df.t1)==False]
   t1=df['t1'].values*50/(1000*60)
    t2=df['t2'].values*50/(1000*60)
    smiles=df['smiles'].values
    rho=df['density g/ml'].values
    V=df['V/ul'].values
    m=rho*V
    V_e=df['loading volume/ul'].values
    e=df['loading solvent'].values
    eluent=df['PE/EA'].values
    speed=df['flow rate ml/min'].values
    eluent_ratio=convert_eluent_ratio(eluent)
    eluent=convert_eluent(eluent)
    e=convert_e(e)
    #np.save('result_save/eluent_ratio.npy',eluent_ratio)
    data={'t1':t1,'t2':t2,'smiles':smiles,'m':m,'V_e':V_e,'e':e,'eluent':eluent,'speed':speed,'eluent_ratio':eluent_ratio}
    return data
def read_data_TLC():
    df=pd.read_excel('TLC_dataset.xlsx')
    ID=df['TLC_ID'].values
    COMPOUND_ID=df['COMPOUND_ID'].values
    H=df['H'].values
    EA=df['EA'].values
    DCM=df['DCM'].values
    MeOH=df['MeOH'].values
    Et2O= df['Et2O'].values
    Rf=df['Rf'].values
    smiles=df['COMPOUND_SMILES'].values
    data = {'TLC_ID': ID, 'COMPOUND_ID': COMPOUND_ID, 'H': H, 'EA': EA, 'DCM': DCM, 'MeOH': MeOH,
            'Et2O': Et2O, 'Rf': Rf,'smiles':smiles}
    return data
def read_data_HPLC():
    df = pd.read_csv('SMRT_dataset.csv',delimiter=';')
    inchi=df['inchi'].values
    RT=df['rt'].values
    data={'inchi':inchi,'RT':RT}
    return data
def mord(mol, nBits=1826, errors_as_zeros=True):
    calc = Calculator(descriptors, ignore_3D=False)
    try:
        result = calc(mol)
        desc_list = [r if not is_missing(r) else 0 for r in result]
        np_arr = np.array(desc_list)
        return np_arr
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def calcu_mord_CC():
    data=read_data_CC()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_CC.npy',all_descriptor)

def calcu_mord_TLC():
    data=read_data_TLC()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_TLC.npy',all_descriptor)


def calcu_mord_CC_8():
    data=read_data_CC_8()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_CC_8.npy',all_descriptor)

def calcu_mord_CC_25():
    data=read_data_CC_25()
    all_descriptor=[]
    for i in tqdm(range(len(data['smiles']))):
        smile = data['smiles'][i]
        mol = Chem.MolFromSmiles(smile)
        descriptor=mord(mol)
        all_descriptor.append(descriptor)
    all_descriptor=np.vstack(all_descriptor)
    np.save('dataset_save/all_descriptor_CC_25.npy',all_descriptor)

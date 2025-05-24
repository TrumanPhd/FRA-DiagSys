#dataloader
#author: guohao wang

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def label():
    labels = list([0] * 9 + [1] * 4 * 9 + [2] * 19 * 9 + [3] * 24 * 9 + [4] * 15 * 9)
    sparse_matrix = np.zeros((len(labels), 5), dtype=int)
    for i, label in enumerate(labels):
        sparse_matrix[i, label] = 1
    return sparse_matrix

def label_degree():
    lab = []
    for i in range(6):
        for j in range(4):
            lab.extend(list([j+9]*9))
            
    label = list([0]*9+[1]*9+[2]*9+[3]*9+[4]*9+[5]*6*9+[6]*6*9+[7]*4*9+[8]*3*9+lab+[13]*10*9+[14]*5*9)
    #print(len(label))
    return label

#label_degree()

def label_every():
    result = []
    for i in range(63):
        for j in range(9):
            result.append(i)
    #print(len(result))
    return result


# for 10bing transformer
def label_degree_EEandTF():
    lab = []
    Fb  = []
    Dsv = []
    for i in range(4):
        lab.extend(list([i+1]*10))
        Fb.extend(list([i+5]*30))
        Dsv.extend(list([i+9]*30))
    label = list([0]*5+lab+Fb+Dsv)        
    return label 


#dataloader for 12bing transformer
class G3T(Dataset):
    def __init__(self):
        df = pd.read_excel('EE12.xlsx')
        #label_mode1
        self.labels = list([0] * 9 + [1] * 4 * 9 + [2] * 19* 9 + [3] * 24 * 9 + [4] * 15 * 9)
        #softmax
        #label_mode2
        #self.labels = label()
        
        df = (df.iloc[4:, :]).T
        #df = df.applymap(lambda x: x * 100000)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


class G3D(Dataset):
    def __init__(self):
        df = pd.read_excel('EE12.xlsx')
        self.labels = label_degree()
        
        df = (df.iloc[4:, :]).T
        #df = df.applymap(lambda x: x * 100000)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

#every_Dataset
class G3E(Dataset):
    def __init__(self):
        df = pd.read_excel('EE12.xlsx')
        self.labels = label_every()
        #softmax
        #label_mode2
        #self.labels = label()
        
        df = (df.iloc[4:, :]).T
        #df = df.applymap(lambda x: x * 100000)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y
    
    
#dataloader for 10bing transformer    
#dataloader for EE winding
class G1T(Dataset):
    def __init__(self):
        df = pd.read_csv('EE.csv')
        #label_mode1
        self.labels = list([0] * 5 + [1] * 4 * 10 + [2] * 12 * 10 + [3] * 12 * 10)
        df = (df.iloc[4:, :]).T
        df = df.astype(np.float32)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


class G1D(Dataset):
    def __init__(self):
        df = pd.read_csv('EE.csv')
        self.labels = label_degree_EEandTF()
        
        df = (df.iloc[4:, :]).T
        df = df.astype(np.float32)
        #df = df.applymap(lambda x: x * 100000)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

#dataloader for TF transformer
class G2T(Dataset):
    def __init__(self):
        df = pd.read_csv('TF.csv')
        #label
        self.labels = list([0] * 5 + [1] * 4 * 10 + [2] * 12 * 10 + [3] * 12 * 10)
        df = (df.iloc[4:, :]).T
        df = df.astype(np.float32)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y


class G2D(Dataset):
    def __init__(self):
        df = pd.read_csv('TF.csv')
        self.labels = label_degree_EEandTF()
        
        df = (df.iloc[4:, :]).T
        df = df.astype(np.float32)
        #df = df.applymap(lambda x: x * 100000)
        self.data = df.values.tolist()
        self.n_samples = len(self.labels)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #'data': torch.cat([self.normal[index], self.AD[index], self.FB[index], self.DSV[index], self.SHORT[index]], dim=1),
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y
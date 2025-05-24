# Auther: Guohao Wang
# Date:   2023/10/13

#fusion
import os
# CUDA_LAUNCH_BLOCKING=1
#subprocess.run(["export", "CUDA_LAUNCH_BLOCKING=1"], shell=True)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataloader import *
from models import *
from model_lib import *
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import argparse

parser = argparse.ArgumentParser(description='Your script description here')

# base
parser.add_argument('--shiyan', type=str, default='G1T', help='Description of shiyan parameter')
parser.add_argument('--train_models', type=int, default=1, help='Description of train or not')
parser.add_argument('--model', type=str, default="FiveClassClassifierWithSoftmax", help='Choose of model')
parser.add_argument('--dropout', type=bool, default=False, help='Choose of dropout')
parser.add_argument('--gpu', type=str, default="2", help='Choose of gpu')

# training
parser.add_argument('--use_tensorboard', type=bool, default=False, help='Choose of use_tensorboard')
parser.add_argument('--use_lr_scheduler', type=bool, default=False, help='Choose of use_lr_scheduler')
parser.add_argument('--use_early_stopping', type=bool, default=False, help='Choose of use_early_stopping')
parser.add_argument('--batch_size', type=int, default=512, help='Choose of batch_size')
parser.add_argument('--num_epochs', type=int, default=500, help='Choose of num_epochs')
#early stopping
parser.add_argument('--early_stopping_patience', type=int, default=200, help='Choose of patience')
parser.add_argument('--early_stopping_counter', type=int, default=200, help='Choose of stopping counter')

args = parser.parse_args()

# config
train_models = bool(args.train_models)
shiyan = args.shiyan
model_ = args.model
model_class = eval(model_)
gpu = args.gpu
dropout = args.dropout
use_tensorboard = args.use_tensorboard
batch_size = args.batch_size
num_epochs = args.num_epochs
early_stopping_patience = args.early_stopping_patience
early_stopping_counter = args.early_stopping_counter
#train_models = bool(1)
#shiyan = 'G3D'

"""
# informations
#Group1 EE10 type     G1T
#num_classes = 4
shiyan = 'G1T'
#Group1 EE10 degree   G1D
#num_classes = 13
shiyan = 'G1D'
#Group2 TF10 type     G2T
#num_classes = 4
shiyan = 'G2T'
#Group2 TF10 degree   G2D
#num_classes = 13
shiyan = 'G2D'
# Group3 EE12 type    G3T
#num_classes = 5
shiyan = 'G3T'
# Group3 EE12 degree  G3D
#num_classes = 14
shiyan = 'G3D'
# G3E
"""

if shiyan == 'G3T':
    num_classes = 5
elif shiyan == 'G3D':
    num_classes = 15
elif shiyan == 'G1T':
    num_classes = 4
elif shiyan == 'G2T':
    num_classes = 4
else:
    num_classes = 13

    
# Dataset
dataset = eval(shiyan)()

#batch_size = 512

# Define the number of folds for cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Create data loaders for training and validation within each fold
data_loaders = []
validation_loaders = []

for train_indices, val_indices in kf.split(dataset):
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    data_loaders.append(train_loader)
    validation_loaders.append(val_loader)

# train
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

device = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")

def trainer(data_loaders,validation_loaders):
    for test_num in range(10):
        data_loader = data_loaders[test_num]
        validation_dataloader = validation_loaders[test_num]
        
        print(f"Fold {test_num + 1}/{n_splits}")
        model = model_class(dropout=dropout,num_classes=num_classes).to(device)
        #num_epochs = 500
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        if use_tensorboard == True:
            training_name = "tesnorboard"+shiyan+"/"+str(test_num)  # 替换为您希望的名称
            writer = SummaryWriter(training_name)

        best_model_params = None
        best_validation_loss = float('inf')
        #early_stopping_patience = 200  # 设定早停的耐心值
        #early_stopping_counter = 200  # 用于计算连续验证损失上升的次数

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for data, labels in data_loader:
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(data)
                #print(outputs)
                #print(outputs.shape)
                #print(labels)
                #print(labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                _, predicted = torch.max(outputs, -1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            epoch_loss = total_loss / len(data_loader)
            epoch_accuracy = correct_predictions / total_samples * 100

            #print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
            if use_tensorboard == True:
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

            model.eval()  
            with torch.no_grad():
                validation_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                for val_data, val_labels in validation_dataloader:
                    val_data, val_labels = val_data.to(device), val_labels.to(device)  # 将验证数据移动到GPU上
                    val_outputs = model(val_data)
                    val_loss = criterion(val_outputs, val_labels)
                    validation_loss += val_loss.item()
                    _, predicted = torch.max(val_outputs, 1)
                    total_samples += val_labels.size(0)
                    correct_predictions += (predicted == val_labels).sum().item()

                validation_loss /= len(validation_dataloader)
                accuracy = (correct_predictions / total_samples) * 100.0

            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, train_accuracy: {epoch_accuracy:.2f}%, Accuracy: {accuracy:.2f}%, Validation Loss: {validation_loss:.4f}")
            
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_params = model.state_dict()
                early_stopping_counter = 0  # 重置早停计数器
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break
            if use_tensorboard == True:
                writer.add_scalar('Loss/validation', validation_loss, epoch)
                writer.add_scalar('Accuracy/validation', accuracy, epoch)
            model.train()

        model.load_state_dict(best_model_params)
        torch.save(model,f'./'+shiyan+'saved_model/fold'+str(test_num)+'_type12.pth')
        if use_tensorboard == True:
            writer.close()

if not os.path.exists('./'+shiyan+'saved_model/'):
    os.makedirs('./'+shiyan+'saved_model/')

if train_models == True:
    trainer(data_loaders,validation_loaders)

#test    
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, auc, accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

sn_list, sp_list, precision_list, recall_list, acc_list, f1_list = [], [], [], [], [], []
y_trues = []
y_preds = []

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
accmax = 0
for i in np.arange(0,1.1,0.1):
    acc_list = []
    for fold in range(n_splits):
        model1 = model_class(num_classes=num_classes)
        model1 = torch.load(f'./'+shiyan+'saved_model/fold'+str(fold)+'_type12.pth')
        
        model2 = FiveClassClassifierWithSoftmax(num_classes=num_classes)
        model2 = torch.load(f'../ai/'+shiyan+'saved_model/fold'+str(fold)+'_type12.pth')
        model1.eval()
        model1.to('cpu')
        model2.eval()
        model2.to('cpu')  
    
        y_true = []
        y_pred = []
        
        val_loader = validation_loaders[fold]
        with torch.no_grad():
            for inputs, labels in val_loader:
                output1 = model1(inputs)
                output2 = model1(inputs)
                output = (i*output1+(1-i)*output2)
                _, predicted = torch.max(output, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        acc_list.append(acc)
    acc = sum(acc_list) / n_splits
    if acc >= accmax:
        print(str(shiyan)+" "+str(i)+" "+str(acc))    
        accmax = acc
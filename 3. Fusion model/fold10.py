# Auther: Guohao Wang
# Date:   2023/10/02

#import subprocess
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
parser.add_argument('--shiyan', type=str, default='G1T', help='Description of shiyan parameter')
parser.add_argument('--train_models', type=int, default=1, help='Description of train or not')
parser.add_argument('--model', type=str, default="FiveClassClassifierWithSoftmax", help='Choose of model')
args = parser.parse_args()

# config
train_models = bool(args.train_models)
shiyan = args.shiyan
model_ = args.model
model_class = eval(model_)

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

batch_size = 512

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

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

def trainer(data_loaders,validation_loaders):
    for test_num in range(10):
        data_loader = data_loaders[test_num]
        validation_dataloader = validation_loaders[test_num]
        
        print(f"Fold {test_num + 1}/{n_splits}")
        model = model_class(dropout=False,num_classes=num_classes).to(device)
        num_epochs = 500
        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 创建TensorBoard的SummaryWriter
        training_name = "tesnorboard"+shiyan+"/"+str(test_num)  # 替换为您希望的名称
        writer = SummaryWriter(training_name)

        best_model_params = None
        best_validation_loss = float('inf')
        early_stopping_patience = 200  # 设定早停的耐心值
        early_stopping_counter = 200  # 用于计算连续验证损失上升的次数

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

            writer.add_scalar('Loss/validation', validation_loss, epoch)
            writer.add_scalar('Accuracy/validation', accuracy, epoch)
            model.train()

        model.load_state_dict(best_model_params)
        torch.save(model,f'./'+shiyan+'saved_model/fold'+str(test_num)+'_type12.pth')
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
for fold in range(n_splits):
    model = model_class(num_classes=num_classes)
    model = torch.load(f'./'+shiyan+'saved_model/fold'+str(fold)+'_type12.pth')
    model.eval()
    model.to('cpu')
    
    y_true = []
    y_pred = []
    
    val_loader = validation_loaders[fold]
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    c_matrix = np.zeros((2, 2), dtype=np.int32)
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            c_matrix[0, 0] += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            c_matrix[0, 1] += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            c_matrix[1, 0] += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            c_matrix[1, 1] += 1

    tn = c_matrix[0, 0]
    fp = c_matrix[0, 1]
    fn = c_matrix[1, 0]
    tp = c_matrix[1, 1]

    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,average='macro')

    sn_list.append(sn)
    sp_list.append(sp)
    precision_list.append(precision)
    recall_list.append(recall)
    acc_list.append(acc)
    f1_list.append(f1)
    y_trues.extend(y_true)
    y_preds.extend(y_pred)

results_df = pd.DataFrame({
    'SN': sn_list,
    'SP': sp_list,
    'Precision': precision_list,
    'Recall': recall_list,
    'Accuracy': acc_list,
    'F1': f1_list
})

avg_sn = sum(sn_list) / n_splits
avg_sp = sum(sp_list) / n_splits
avg_precision = sum(precision_list) / n_splits
avg_recall = sum(recall_list) / n_splits
avg_acc = sum(acc_list) / n_splits
avg_f1_score = sum(f1_list) / n_splits

std_sn = (sum([(sn - avg_sn) ** 2 for sn in sn_list]) / (n_splits - 1)) ** 0.5
std_sp = (sum([(sp - avg_sp) ** 2 for sp in sp_list]) / (n_splits - 1)) ** 0.5
std_precision = (sum([(precision - avg_precision) ** 2 for precision in precision_list]) / (n_splits - 1)) ** 0.5
std_recall = (sum([(recall - avg_recall) ** 2 for recall in recall_list]) / (n_splits - 1)) ** 0.5
std_acc = (sum([(acc - avg_acc) ** 2 for acc in acc_list]) / (n_splits - 1)) ** 0.5
std_f1_score = (sum([(f1 - avg_f1_score) ** 2 for f1 in f1_list]) / (n_splits - 1)) ** 0.5

# save
results_df.loc["Fold Average"] = [avg_sn, avg_sp, avg_precision, avg_recall, avg_acc, avg_f1_score]
results_df.loc["Std Deviation"] = [std_sn, std_sp, std_precision, std_recall, std_acc, std_f1_score]
results_df.to_excel("evaluation/"+shiyan+"_results.xlsx")



def plot_confusion_matrix(y_true, y_pred, classes, save_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred)
    #print(len(y_pred))
    #print(len(y_true))
    #print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    if normalize:
        plt.savefig(save_path, dpi=300)
    else:
        plt.savefig(save_path, dpi=300)

    plt.show()

def gene_name(str_i):
    str_ = []
    for i in range(1,5):
        str_.append(str_i+str(i))
    return str_    
    
if shiyan == 'G1T' or shiyan == 'G2T':
    class_names = np.array(["Normal", "AD", "FB", "DSV"])
elif shiyan == 'G1D' or shiyan == 'G2D':
    class_names = np.array(["Normal"]+gene_name("AD")+gene_name("FB")+gene_name("DSV")) 
elif shiyan == 'G3T':
    class_names = np.array(["Normal", "AD", "FB", "DSV", "SC"])
else:
    class_names = np.array(["Normal"]+gene_name("AD")+gene_name("FB")+gene_name("DSV")+["SC1"]+["SC2"]) 

plot_confusion_matrix(y_trues, y_preds, classes=class_names, save_path='figures/'+shiyan+'_confusion_matrix.TIF', normalize=True)
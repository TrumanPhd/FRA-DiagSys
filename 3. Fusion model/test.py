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
print(model_class)
print("---------")
model_real = model_class(num_classes=13) 
print(model_real)
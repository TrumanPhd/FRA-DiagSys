# models for transformer disable check
# author: Guohao Wang
# Date: 2023


import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#from MultiRM.Scripts.models import *
from util_layers import *

# ai
class ly3(nn.Module):
    def __init__(self, num_classes=5,dropout=False):
        super(ly3, self).__init__()
        self.fc1 = nn.Linear(2000, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, num_classes)
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = F.log_softmax(out, dim=-1)
        return out


class ly7(nn.Module):
    def __init__(self, num_classes=5,dropout=False):
        super(ly7, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4000, 8000)
        self.fc3 = nn.Linear(8000, 4000)
        self.fc4 = nn.Linear(4000, 2000)
        self.fc5 = nn.Linear(2000, 1000)
        self.fc6 = nn.Linear(1000, 500)
        self.fc7 = nn.Linear(500, num_classes)
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        if self.drop_state == True:
            out = self.dropout(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = F.log_softmax(out, dim=-1)
        return out

# ai
class FiveClassClassifierWithSoftmax(nn.Module):
    def __init__(self, num_classes=5,dropout=False):
        super(FiveClassClassifierWithSoftmax, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)
        self.fc3 = nn.Linear(4000, 2000)
        self.fc4 = nn.Linear(2000, 1000)
        self.fc5 = nn.Linear(1000, 500)
        #p the probality to be cut
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        if self.drop_state == True:
            out = self.dropout(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        if self.drop_state == True:
            out = self.dropout(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=-1)
        return out

# demo2
class demo2(nn.Module):
    def __init__(self, num_classes=5, dropout=False):
        super(demo2, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4000, 3000)
        self.fc3 = nn.Linear(3000, 2000)
        self.fc4 = nn.Linear(2000, 1500)
        self.fc5 = nn.Linear(1500, 1000)
        self.fc6 = nn.Linear(1000, 800)
        self.fc7 = nn.Linear(800, 600)
        self.fc8 = nn.Linear(600, 400)
        self.fc9 = nn.Linear(400, 200)
        self.fc10 = nn.Linear(200, num_classes)
        
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        if self.drop_state:
            out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        if self.drop_state:
            out = self.dropout(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        out = self.relu(out)
        out = self.fc10(out)
        out = F.log_softmax(out, dim=-1)
        return out

# ai2
class demo3(nn.Module):
    def __init__(self, num_classes=5, dropout=False):
        super(demo3, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.fc2 = nn.Linear(4000, 8000)  
        self.fc3 = nn.Linear(8000, 16000)  
        self.fc4 = nn.Linear(16000, 8000)  
        self.fc5 = nn.Linear(8000, 4000)
        self.fc6 = nn.Linear(4000, 2000)
        self.fc7 = nn.Linear(2000, 1000)
        self.fc8 = nn.Linear(1000, 500)
        self.fc9 = nn.Linear(500, 250)
        self.fc10 = nn.Linear(250, num_classes)
        
        self.relu = nn.ReLU()
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        if self.drop_state:
            out = self.dropout(out)
        out = self.fc5(out)
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.fc8(out)
        out = self.relu(out)
        out = self.fc9(out)
        out = self.relu(out)
        out = self.fc10(out)
        out = F.log_softmax(out, dim=-1)
        return out

# ai3
class ReducedLayerWideClassifierWithAttention(nn.Module):
    def __init__(self, num_classes=5, dropout=True):
        super(ReducedLayerWideClassifierWithAttention, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.fc2 = nn.Linear(4000, 2000)
        self.fc3 = nn.Linear(2000, 1000)
        
        # 添加多头自注意力机制
        self.multihead_attention = nn.MultiheadAttention(embed_dim=1000, num_heads=4)
        
        self.fc4 = nn.Linear(1000, 500)
        self.fc5 = nn.Linear(500, num_classes)
        
        self.relu = nn.ReLU()
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        
        # 添加多头自注意力机制
        if self.drop_state:
            out = self.dropout(out)
        out = out.permute(1, 0, 2)
        attention_output, _ = self.multihead_attention(out, out, out)
        out = attention_output.permute(1, 0, 2)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        if self.drop_state:
            out = self.dropout(out)
        
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        out = F.log_softmax(out, dim=-1)
        return out

# ai4
class ModelWithLSTMAndAttention(nn.Module):
    def __init__(self, input_dim=2000, hidden_dim=256, num_layers=2, num_heads=4, num_classes=5,dropout=True):
        super(ModelWithLSTMAndAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # LSTM层处理序列数据
        lstm_out, _ = self.lstm(x)
        
        # 多头自注意力机制处理LSTM的输出
        lstm_out = lstm_out.permute(1, 0, 2)
        attention_output, _ = self.multihead_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = attention_output.permute(1, 0, 2)
        
        # 线性层和激活函数
        out = self.fc1(lstm_out[:, -1, :])  # 取LSTM输出的最后一个时间步
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=-1)
        return out

"""
# ai5
#import torchvision.transforms as transforms

class AdvancedModel(nn.Module):
    def __init__(self, num_classes=5):
        super(AdvancedModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm = nn.BatchNorm2d(64)  # 批归一化层
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4)  # Transformer层
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)  # 堆叠Transformer层
        self.data_augmentation = nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        )

    def forward(self, x):
        x = self.data_augmentation(x)  # 数据增强
        x = self.conv1(x)
        x = self.batch_norm(x)  # 批归一化
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch_norm(x)  # 批归一化
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x.unsqueeze(0))  # Transformer编码
        x = x.squeeze(0)
        x = self.fc2(x)
        return x


def tester(model):
    input_data = torch.randn(2,2000)
    output = model(input_data)
    print("Input size:", input_data.size())
    print("Output size:", output.size())

#model = FiveClassClassifierWithBiLSTM()
#tester(model)

model1 = FiveClassClassifierWithSoftmax().to('cuda:0')
import torchsummary
model2 = demo2().to('cuda:0')
model3 = demo2(dropout=True).to('cuda:0')
torchsummary.summary(model1, (1,2000))
torchsummary.summary(model2, (1,2000))
torchsummary.summary(model3, (1,2000))
"""

def tester(model):
    input_data = torch.randn(2,2000)
    output = model(input_data)
    print("Input size:", input_data.size())
    print("Output size:", output.size())

#model = ly7(dropout=True)
#tester(model)
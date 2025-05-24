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

#linear model
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



#naive transformer
class TransformerClassifier(nn.Module):
    def __init__(self, out_size=5, num_layers=8, hidden_dim=2000, num_heads=8):
        super(TransformerClassifier, self).__init__() 
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=2000,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
            ),
            num_layers=num_layers,
        )
        self.fc1 = nn.Linear(hidden_dim, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.transformer(x)
        #x = x.mean(dim=1)  # Pooling operation, you can change this as needed
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=-1)
        return x




# complex_net
class complex_net(nn.Module):
    def __init__(self, label_dim=5):
        super(complex_net, self).__init__()
        self.label_dim = label_dim
        self.model = nn.Sequential(
            LearnableMatrix(),
            nn.ReLU(),
            nn.Conv1d(in_channels=2),
            nn.ReLU(),
            #nn.Linear(dim2, dim3),
            #nn.ReLU(),
            #nn.Linear(dim3, dim4),
            #nn.ReLU(),
            #nn.Linear(dim4, dim5),
            #nn.ReLU(),
            #nn.Linear(dim5, dim6),
            nn.ReLU(),
            nn.Linear(516, self.label_dim),
            nn.Softmax(dim=-1) 
        )
        
        

    def forward(self, x):
        return self.model(x)




class FiveClassClassifierWithMultiheadAttention(nn.Module):
    def __init__(self, num_classes=5, dropout=False, num_heads=4, hidden_dim=2000):
        super(FiveClassClassifierWithMultiheadAttention, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)
        self.fc3 = nn.Linear(4000, 2000)
        self.fc4 = nn.Linear(2000, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.fc6 = nn.Linear(4000, 500)
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # 添加多头注意力机制层
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.attention_output_fc = nn.Linear(hidden_dim, 4000)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        if self.drop_state:
            out = self.dropout(out)
        out = self.relu(out)
        
        # 应用多头注意力机制
        attention_output, _ = self.multihead_attention(out, out, out)
        out = self.attention_output_fc(attention_output)
        
        out = self.relu(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=-1)
        return out

class FiveClassClassifierWithBiLSTM(nn.Module):
    def __init__(self, num_classes=5, dropout=False, hidden_size=256, num_layers=2):
        super(FiveClassClassifierWithBiLSTM, self).__init__()
        self.fc1 = nn.Linear(2000, 4000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        self.fc3 = nn.Linear(4000, 2000)
        self.fc4 = nn.Linear(2000, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.drop_state = dropout
        self.dropout = nn.Dropout(p=0.2)
        
        # 添加BiLSTM层
        self.lstm = nn.LSTM(input_size=500, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size * 2  # 因为是双向LSTM，所以隐藏状态的维度需要乘以2

    def forward(self, x):
        print(x.shape)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        print(out.shape)
        if self.drop_state:
            out = self.dropout(out)
        out = self.relu(out)
        
        # 将输出传递给BiLSTM层
        out, _ = self.lstm(out)
        
        # 取最后一个时间步的输出作为分类输入
        out = out[:, :]
        print(out.shape)
        out = self.fc2(out)
        out = F.log_softmax(out, dim=-1)
        return out




#tansformer based model
# import from MultiRM

#


def tester(model):
    input_data = torch.randn(2,2000)
    output = model(input_data)
    print("Input size:", input_data.size())
    print("Output size:", output.size())

#model = FiveClassClassifierWithBiLSTM()
#tester(model)

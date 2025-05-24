#main for the disable check for transformer
#author: Guohao Wang 
#date:   2023 09 13

import pandas as pd
import numpy as np
from model_lib import *
from dataloader import *
from models import *
NUM_EPOCH = 20


#load data 1,2,3,4 is the label for the 4 problem of the transfomer
#          0 is the label for the original transformer
"""
class raw_load():
    def __init__(self):
        self.

def raw_loader(path='EE12.xlsx'):
    
"""
path='EE12.xlsx'
df = pd.read_excel(path)
"""
normal = df.iloc[5:, :9]
################### 
AD1 = df.iloc[5:, 9:18]
AD2 = df.iloc[5:, 18:27]
AD3 = df.iloc[5:, 27:36]
AD4 = df.iloc[5:, 36:45]
AD  = df.iloc[5:, 9:45]

###################
FB1  = df.iloc[5:, 45:54] 
FB3  = df.iloc[5:, 54:63]
FB5  = df.iloc[5:, 63:72]
FB7  = df.iloc[5:, 72:81]
FB9  = df.iloc[5:, 81:90]
FB11 = df.iloc[5:, 90:99]

FB1_2 = df.iloc[5:, 99:108]
FB3_4 = df.iloc[5:, 108:117]
FB5_6 = df.iloc[5:, 117:126]
FB7_8 = df.iloc[5:, 126:135]
FB9_10 = df.iloc[5:, 135:144]
FB11_12 = df.iloc[5:, 144:153]

FB1_2_3 = df.iloc[5:, 153:162]
FB4_5_6 = df.iloc[5:, 162:171]
FB7_8_9 = df.iloc[5:, 171:180]
FB10_11_12 = df.iloc[5:, 180:189]
    
FB1_2_3_4 = df.iloc[5:, 189:198] 
FB5_6_7_8 = df.iloc[5:, 198:207]
FB9_10_11_12 = df.iloc[5:, 207:216]

FB = df.iloc[5:, 45:216]

###################
DSV2_1 = df.iloc[5:, 216:225]
DSV2_2 = df.iloc[5:, 225:234]
DSV2_3 = df.iloc[5:, 234:243]
DSV2_4 = df.iloc[5:, 243:252]

DSV3_1 = df.iloc[5:, 252:261]
DSV3_2 = df.iloc[5:, 261:270]
DSV3_3 = df.iloc[5:, 270:279]
DSV3_4 = df.iloc[5:, 279:288]

DSV6_1 = df.iloc[5:, 288:297]
DSV6_2 = df.iloc[5:, 297:306]
DSV6_3 = df.iloc[5:, 306:315]
DSV6_4 = df.iloc[5:, 315:324]

DSV7_1 = df.iloc[5:, 324:333]
DSV7_2 = df.iloc[5:, 333:342]
DSV7_3 = df.iloc[5:, 342:351]
DSV7_4 = df.iloc[5:, 351:360]

DSV10_1 = df.iloc[5:, 360:369]
DSV10_2 = df.iloc[5:, 369:378]
DSV10_3 = df.iloc[5:, 378:387]
DSV10_4 = df.iloc[5:, 387:396]

DSV11_1 = df.iloc[5:, 396:405]
DSV11_2 = df.iloc[5:, 405:414]
DSV11_3 = df.iloc[5:, 414:423]
DSV11_4 = df.iloc[5:, 423:432]

DSV = df.iloc[5:, 216:432]

ALL = df.iloc[5:, :]

AD  = df.iloc[5:, 9:45]
FB = df.iloc[5:, 45:216]
DSV = df.iloc[5:, 216:432]
SHORT = df.iloc[5:, 432:]


###################
for i in range(10):
    exec(f"SHORT_{i+2} = df.iloc[5:, {432+9*i}:{441+9*i}]")

for i in range(5):
    exec(f"SHORT_{2*i+2}_{2*i+3} = df.iloc[5:, {522+9*i}:{531+9*i}]")

SHORT = df.iloc[5:, 432:]
"""

#processing
custom_dataset = CustomDataset(df)
batch_size = 32  
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

#models
INPUT_DIM = 2000  
HIDDEN_DIM = 256 
OUTPUT_DIM = 5  
NUM_LAYERS = 2  # number of Transformer layers
NUM_HEADS = 8   
DROPOUT = 0.2 
#model = TransformerClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS, NUM_HEADS, DROPOUT)

#model = LinearRegression()

model = NaiveNet_v1()

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from train import *
"""
def train_model(model, dataloader, loss_fn, optimizer, num_epochs=NUM_EPOCH, save_path='model.pth'):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(inputs)
            print(inputs.shape)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100.0
        print(f'Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.2f}%')

    torch.save(model.state_dict(), save_path)
    print('Model saved to', save_path)
"""

# model = YourModel()
# dataloader = YourDataLoader()
# loss_fn = YourLossFunction()
# optimizer = optim.YourOptimizer(model.parameters(), lr=0.001)
#train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
train_model(model, data_loader, loss_fn, optimizer, num_epochs=100, save_path='model.pth')

"""
#test_data =  

#print(model.predict())

torch.save(model.state_dict(), 'transformer_model.pth')

model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()  # 设置为评估模式 不使用Dropout等

with torch.no_grad():
    test_output = model(test_data)
    _, predicted = torch.max(test_output, 1)  # 预测类别

print("预测结果:", predicted)  
"""  
#light try
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataloader import *
from models import *
from model_lib import *
from torch.utils.data import DataLoader, random_split

import os

batch_size = 512

# 设置环境变量
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
input_size = 2000
num_samples = 1000
num_classes = 5

path='EE12.xlsx'
df = pd.read_excel(path)
#custom_dataset = CustomDataset(df)
#data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

#should be changed here
#validation_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Assuming you have already instantiated your CustomDataset with your DataFrame (df)
dataset = CustomDataset(df)

# Define the size for the validation dataset (e.g., 20% of the entire dataset)
validation_size = int(0.1 * len(dataset))
train_size = len(dataset) - validation_size

# Split the dataset into training and validation sets
train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])


# Create data loaders for training and validation
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        

# train
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

for test_num in range(4):
    ###################################################################################
    #model = demo1().to(device)
    model = FiveClassClassifierWithSoftmax(dropout=False).to(device)
    #model = TransformerClassifier().to(device)
    #model = FiveClassClassifierWithMultiheadAttention().to(device)
    #model = FiveClassClassifierWithBiLSTM().to(device)
    num_epochs = 500


    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    ####################################################################################


    # 创建TensorBoard的SummaryWriter
    training_name = r"type/Fault Type Recognition "+str(test_num)  # 替换为您希望的名称
    writer = SummaryWriter(training_name)

    # 初始化变量以跟踪最佳模型参数和验证损失
    best_model_params = None
    best_validation_loss = float('inf')
    early_stopping_patience = 200  # 设定早停的耐心值
    early_stopping_counter = 200  # 用于计算连续验证损失上升的次数

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)  # 将数据移动到GPU上
            optimizer.zero_grad()
            outputs = model(data)
            #print(outputs)
            #print(outputs.shape)
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

        # 将损失和准确率写入TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

        # 验证模型并检查早停条件

        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            validation_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for val_data, val_labels in validation_dataloader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)  # 将验证数据移动到GPU上
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_labels)
                validation_loss += val_loss.item()

                # 计算准确率
                _, predicted = torch.max(val_outputs, 1)
                total_samples += val_labels.size(0)
                correct_predictions += (predicted == val_labels).sum().item()

            validation_loss /= len(validation_dataloader)
            accuracy = (correct_predictions / total_samples) * 100.0

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, train_accuracy: {epoch_accuracy:.2f}%, Accuracy: {accuracy:.2f}%, Validation Loss: {validation_loss:.4f}")
        
        if validation_loss < best_validation_loss:
            # 保存当前最佳模型参数
            best_validation_loss = validation_loss
            best_model_params = model.state_dict()
            early_stopping_counter = 0  # 重置早停计数器
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break

        # 将验证损失写入TensorBoard
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)
        model.train()  # 将模型恢复为训练模式

    # 在训练结束后，恢复使用最佳模型参数
    model.load_state_dict(best_model_params)

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    #model.predict()
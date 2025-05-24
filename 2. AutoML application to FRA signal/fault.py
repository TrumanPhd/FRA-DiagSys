# autogluon
# Guohao Wang

from f5C_utils  import *
from dataloader import *
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import torch
from sklearn.model_selection import train_test_split

# 读取数据集
df = pd.read_excel('EE12.xlsx')

# 转置数据框
df = df.iloc[4:, :].T

# 创建TabularDataset对象
dataset = TabularDataset(df)

# 创建标签列
dataset['label'] = label_degree()

# 保存TabularDataset对象为CSV文件（AutoGluon要求数据保存为CSV）
dataset.to_csv('autogluon_dataset.csv')

# 指定训练数据文件
train_data = 'autogluon_dataset.csv'

# 定义标签列名
label = 'label'

# 分割数据集为训练集和测试集（10%作为测试集）
train_df, test_df = train_test_split(dataset, test_size=0.1, random_state=42)

# 保存训练集和测试集为CSV文件
train_df.to_csv('train_data.csv')
test_df.to_csv('test_data.csv')

# 创建TabularPredictor并进行训练
predictor = TabularPredictor(label=label, path='autogluon_models').fit(train_data, time_limit=600, presets='best_quality')#, use_gpu=True)

# 进行测试
test_data = TabularDataset('test_data.csv')  # 使用测试集数据
scores = predictor.evaluate(test_data)

# 打印测试预测结果
print(scores)



# Rf
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


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
def G3D():
    df = pd.read_excel('EE12.xlsx')
    labels = label_degree()
    
    df = (df.iloc[4:, :]).T
    #df = df.applymap(lambda x: x * 100000)
    data = df.values.tolist()
    return data, labels
        
    
    
#dataloader for 10bing transformer    
#dataloader for EE winding
def G1T():
    df = pd.read_csv('EE.csv')
    #label_mode1
    labels = list([0] * 5 + [1] * 4 * 10 + [2] * 12 * 10 + [3] * 12 * 10)
    df = (df.iloc[4:, :]).T
    df = df.astype(np.float32)
    data = df.values.tolist()
    return data, labels
        


def G1D():
    df = pd.read_csv('EE.csv')
    labels = label_degree_EEandTF()
    df = (df.iloc[4:, :]).T
    df = df.astype(np.float32)
    data = df.values.tolist()
    return data, labels
        

#dataloader for TF transformer
def G2T():
    df = pd.read_csv('TF.csv')
    #label
    labels = list([0] * 5 + [1] * 4 * 10 + [2] * 12 * 10 + [3] * 12 * 10)
    df = (df.iloc[4:, :]).T
    df = df.astype(np.float32)
    data = df.values.tolist()
    return data, labels


def G2D():
    df = pd.read_csv('TF.csv')
    labels = label_degree_EEandTF()
    df = (df.iloc[4:, :]).T
    df = df.astype(np.float32)
    data = df.values.tolist()
    return data, labels    

def G3T():
    df = pd.read_excel('EE12.xlsx')
    #label_mode1
    y = list([0] * 9 + [1] * 4 * 9 + [2] * 19* 9 + [3] * 24 * 9 + [4] * 15 * 9)    
    df = (df.iloc[4:, :]).T
    X = df.values.tolist()
    return X, y

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# 定义一个函数用于加载不同的数据集
def load_data(dataset):
    if dataset == "G1T":
        # Load G1T data
        # Replace this with your data loading logic for G1T
        X, y = G1T()
        return X, y
    elif dataset == "G1D":
        # Load G1D data
        # Replace this with your data loading logic for G1D
        X, y = G1D()
        return X, y
    elif dataset == "G2T":
        # Load G2T data
        # Replace this with your data loading logic for G2T
        X, y = G1T()
        return X, y
    elif dataset == "G2D":
        # Load G2D data
        # Replace this with your data loading logic for G2D
        X, y = G2D()
        return X, y
    elif dataset == "G3T":
        # Load G3T data
        # Replace this with your data loading logic for G3T
        X, y = G3T()
        return X, y
    elif dataset == "G3D":
        # Load G3D data
        # Replace this with your data loading logic for G3D
        X, y = G3D()
        return X, y
    else:
        return None, None

# 创建一个空的DataFrame来存储实验结果
results_df = pd.DataFrame(columns=["Dataset", "Metric", "Mean", "Std Error"])

# 定义六个不同的数据集
datasets = ["G1T", "G1D", "G2T", "G2D", "G3T", "G3D"]

# 循环遍历六个数据集并进行实验
for dataset in datasets:
    # 加载数据集
    X, y = load_data(dataset)
    
    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义随机森林分类器
    rf_classifier = RandomForestClassifier()

    # 定义超参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 创建GridSearchCV对象，进行网格搜索和交叉验证
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=10, n_jobs=-1)

    # 在训练集上拟合模型
    grid_search.fit(X_train, y_train)

    # 使用最佳参数的模型进行预测
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)


    # 计算模型性能指标
    accuracy = cross_val_score(best_rf_model, X, y, cv=10, scoring='accuracy')
    precision_micro = cross_val_score(best_rf_model, X, y, cv=10, scoring='precision_micro')
    recall_micro = cross_val_score(best_rf_model, X, y, cv=10, scoring='recall_micro')
    f1_micro = cross_val_score(best_rf_model, X, y, cv=10, scoring='f1_micro')

    # 计算混淆矩阵
    #conf_matrix = confusion_matrix(y, y_pred)

    # 计算敏感性（SN）和特异性（SP）
    #sensitivity = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    #specificity = 1 - (conf_matrix.sum(axis=0) - conf_matrix.diagonal()) / (conf_matrix.sum() - conf_matrix.sum(axis=1))

    # 计算标准误差
    std_error_accuracy = np.std(accuracy)
    std_error_precision_micro = np.std(precision_micro)
    std_error_recall_micro = np.std(recall_micro)
    std_error_f1_micro = np.std(f1_micro)
    #std_error_sensitivity = np.std(sensitivity)
    #std_error_specificity = np.std(specificity)

    # 将实验结果添加到临时DataFrame中
    temp_df = pd.DataFrame({
        "Dataset": dataset,
        "Metric": ["Accuracy", "Precision (Micro)", "Recall (Micro)", "F1-score (Micro)"],# "Sensitivity (SN)", "Specificity (SP)"],
        "Mean": [accuracy.mean(), precision_micro.mean(), recall_micro.mean(), f1_micro.mean()],# sensitivity.mean(), specificity.mean()],
        "Std Error": [std_error_accuracy, std_error_precision_micro, std_error_recall_micro, std_error_f1_micro],# std_error_sensitivity, std_error_specificity]
    })

    # 将临时DataFrame添加到结果DataFrame中
    results_df = pd.concat([results_df, temp_df], ignore_index=True)

# 将结果保存到CSV文件
results_df.to_csv("experiment_results.csv", index=False)

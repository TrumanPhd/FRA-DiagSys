# SVM
import sklearn 
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
import pandas as pd

def load_your_data():
    df = pd.read_excel('EE12.xlsx')
    #label_mode1
    y = list([0] * 9 + [1] * 4 * 9 + [2] * 19* 9 + [3] * 24 * 9 + [4] * 15 * 9)
    #softmax
    #label_mode2
    #self.labels = label()
    
    df = (df.iloc[4:, :]).T
    X = df.values.tolist()
    return X, y

# 假设你已经有了数据集 X 和标签 y
X, y = load_your_data()

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义SVM模型
svm_model = SVC()

# 定义参数网格
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}

# 创建GridSearchCV对象，进行网格搜索和交叉验证
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=10, n_jobs=-1)

# 在训练集上拟合模型
grid_search.fit(X_train, y_train)

# 打印最佳参数
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳参数的模型进行预测
best_svm_model = grid_search.best_estimator_
y_pred = best_svm_model.predict(X_test)

# 计算模型性能
accuracy = cross_val_score(best_svm_model, X, y, cv=10)
print("Accuracy: {:.2f}%".format(accuracy.mean() * 100))

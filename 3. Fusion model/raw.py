#raw 
import pandas as pd

"""
# 读取CSV文件
df = pd.read_excel('EE12.xlsx')

# 删除第6行到最后一行的所有负号
df.iloc[5:, :] = df.iloc[5:, :].apply(lambda x: x.str.replace('-', ''))

# 将处理后的数据保存回CSV文件
df.to_csv('EE12.csv', index=False)
"""


# 读取CSV文件
df = pd.read_excel('EE.xlsx').astype(str)

# 删除第6行到最后一行的所有负号
df.iloc[5:, :] = df.iloc[5:, :].apply(lambda x: x.str.replace('-', ''))

# 将处理后的数据保存回CSV文件
df.to_csv('EE.csv', index=False)

df = pd.read_excel('TF.xlsx').astype(str)

# 删除第6行到最后一行的所有负号
df.iloc[5:, :] = df.iloc[5:, :].apply(lambda x: x.str.replace('-', ''))

# 将处理后的数据保存回CSV文件
df.to_csv('TF.csv', index=False)

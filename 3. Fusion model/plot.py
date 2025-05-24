# plot the FRA curve
# Author: Guohao Wang
# Date: 2023/10/20
import pandas as pd
import matplotlib.pyplot as plt

di = 15
# 读取数据
df = pd.read_csv('TF.csv')

# 选择要可视化的数据列，替换id和id+1为你的实际列索引
id = 0  # 请替换为你的列索引
data = df.iloc[4:, id:id+1].T

# 提取横轴和纵轴数据
#frequency = data['Frequency/kHz']
#magnitude = data['Magnitude/db']
freq = list(i for i in range(1,2001))
magnitude = list(data)
# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(freq, magnitude, label='曲线1', color='b')  # 曲线1的标注和颜色可以自行更改

# 添加标签和标题
plt.xlabel('Frequency/kHz')
plt.ylabel('Magnitude/db')
plt.title('数据可视化')

# 添加图例
plt.legend()

# 显示图表
plt.grid(True)
plt.savefig('TF.TIF', dpi=300)

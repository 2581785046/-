# ① 导入相关库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# ② 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ③ 读取数据（假设文件位于当前目录）
df_density = pd.read_csv(r'C:/Users/25817/PycharmProjects/pythonProject/kaozheng/素材-001/素材/人口密度数据.csv', encoding='gb18030')
df_traffic = pd.read_csv('C:/Users/25817/PycharmProjects/pythonProject/kaozheng/素材-001/素材/交通拥堵数据.csv', encoding='gb18030')

# 合并数据集（假设通过区域列合并）
df = pd.merge(df_density, df_traffic, on='地区')

# ④ 绘制散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['人口密度（人/平方米）'], df['交通拥堵指数'], alpha=0.6)
plt.xlabel('人口密度（人/km²）')
plt.ylabel('交通拥堵指数')
plt.title('人口密度与交通拥堵指数关系')
plt.grid(True)
plt.show()

# ⑤ 线性回归分析
X = df[['地区']]
y = df['交通拥堵指数']

model = LinearRegression()
model.fit(X, y)

# 计算指标
correlation = df['人口密度'].corr(df['交通拥堵指数'])
r_squared = model.score(X, y)

# ⑥ 计算每增加1000人/km²的影响
coef = model.coef_[0]
increase_per_1000 = coef * 1000

# ⑦ 预测3500人/km²的拥堵指数
prediction = model.predict([[3500]])[0]

print(f"相关系数: {correlation:.3f}")
print(f"决定系数: {r_squared:.3f}")
print(f"人口密度每增加1000人/km²，拥堵指数增加: {increase_per_1000:.3f}")
print(f"预测拥堵指数: {prediction:.3f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv('./data/智慧农业数据.csv', encoding='gb18030')
# 计算温度、湿度和光照列的均值和标准差
mean_temperature = round(df['温度（摄氏度）'].mean(), 2)
std_temperature = round(df['温度（摄氏度）'].std(), 2)
mean_humidity = round(df['湿度（%）'].mean(), 2)
std_humidity = round(df['湿度（%）'].std(), 2)
mean_light = round(df['光照（小时）'].mean(), 2)
std_light = round(df['光照（小时）'].std(), 2)
# 输出统计分析结果
print("温度：均值={}, 标准差={}".format(mean_temperature, std_temperature))
print("湿度：均值={}, 标准差={}".format(mean_humidity, std_humidity))
print("光照：均值={}, 标准差={}".format(mean_light, std_light))
# 定义自变量和因变量
X = df[['温度（摄氏度）', '湿度（%）', '光照（小时）']]
y = df['产量（吨）']
# 创建线性回归模型并拟合数据
model = LinearRegression()
model.fit(X, y)
# 输出回归模型的系数和截距
coefficients = model.coef_
intercept = model.intercept_
print("回归系数：")
print(coefficients)
print("截距：")
print(intercept)

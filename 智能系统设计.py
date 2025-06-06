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

#
# 根据统计分析的结果，可以了解农作物生长环境的温度、湿度和光照的整体情况。
# 温度分析：观察温度的平均值和标准差，可以判断环境的温度水平和波动程度。在适宜的温度范围
# 内，农作物通常能够良好生长。在这组数据中，温度的平均值为26.2，符合农作物两盒生长的适宜温
# 度范围，标准差为3.03，标准差比较大，说明环境相对适宜到温度波动较大。根据目标作物的需求，
# 可以进一步评估环境是否适宜生长。
# 湿度分析：湿度的均值和标准差可以揭示环境湿度的平均水平和变化程度。不同的农作物对湿度有不
# 同的要求。湿度的标准差为5.59,标准差较大，说明环境湿度波动较大。对于特定的作物，可以根据其
# 生长习性来判断湿度情况是否适宜。
# 光照分析：光照的均值和标准差可以反映环境的光照水平和变化程度。光照是植物进行光合作用的重
# 要因素，对于植物的生长和发育至关重要。比较光照的均值和标准差可以判断环境光照的平均水平和
# 光照波动的幅度。光照平均值为7.0，标准差为1.58，属于光照的平均值较高但标准差较大的情况，说
# 明环境光照充足但波动较大。
# 通过回归分析，建立了温度、湿度和光照与产量之间的线性回归模型。根据模型的系数，可以得知湿
# 度和光照对产量有正向影响，而温度对产量有负向影响。但是，这个分析结果仅基于提供的样本数
# 据，可能并不能代表智慧农业中所有农作物的生长情况。


# 数据集的三种划分方法培训大纲
# 培训课时
# 20课时
# 培训内容
# 数据集划分概述
# 留出法概述与使用
# 交叉验证法概述与使用
# 交差验证法的实现过程
# 自助法的概述与使用
# 三种方法的优缺点

#导入相关库
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#读取“某路口的交通流量监测数据.csv”文件
data=pd.read.scv('./某路口的交通流量监测数据.scv')

#提取字段为“HR”“WEEK_DAY""DAY OF YEAR""WEEK OF YEAR"的数据内容作为数据特征，字段"TRAFFIC_COUNT"的数据内容作为数据标签
feature=data.lioc[:,1:5]
y=data['TRAFFIC_COUNT']

#构建特征。使用PolynomialF-eatures函数创建最高次数为6次方的多项式特征
ploy=PolynomialFeatures(6)
X=ploy.fit_transform(feature)

#构建岭回归模型，并使用创建的多项式特征进行模型训练
rid=Ridge(alpha=20.0,fit intercept=True))
rid.fit(X,y)

#绘制真实结果和预测结果的拟合曲线（需设置X轴取值范围为200到300），从而直观的查看模型的拟合效果
start=200 
end= 300
y pre = rid.predict(X)
time=np.arange(start, end)
plt.plot(time, y[start:end], 'b', label="real")
plt.plot(time,y pre[start:end],r,label='predict)
plt.legend(loc='upper left')
plt.show()

#对模型进行评估，主要为计算并打印模型的R方值
r2=r2_score(y,y_pre,multioutput='raw_values')
print(‘R方值:’,r2)

#对模型评估结果进行分析
从我们的拟合曲线可以看出，整体上模型预测与真实趋势比较接近，R方为0.69，整体模型效果的误差在30%，整体效果较好，但还有优化的空间。

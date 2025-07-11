#导入相关库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#将字体设置为中文
plt.rcParams['font.sans-serif']='SimHei' #黑体

#数据加载
data= pd.read csv(‘./国民经济核算季度数据.csv’,encoding='gbk')

#查看第一、二、三产业国内生产总值当季值的时间变化趋势
plt.plot(np.array(data[‘时间’]), np.array(data['第一产业增加值_当季值(亿元)’]))
plt.plot(np.array(data['时间’),np.array(data['第二产业增加值_当季值(亿元)’]))
plt.plot(np.array(data['时间'),np.array(data['第三产业增加值_当季值(亿元)’]))
plt.legend([data.columns[i][:4]for i in [2,3,4]])#添加图例(优先plot，按顺序)
plt.xticks(np.array(data[‘时间’)[::4],rotation=90)#将x坐标轴上的标签旋转90度
plt.xlabel('时间')
plt.ylabel(国内生产总值_当季值(亿元)’)
plt.title(‘国内生产总值当季值随时间变化趋势情况’)
plt.show()

#比较2017年第一季度第一二三产业的生产总值数值情况
num= data.iloc[-1.5:].sort_values()
label = [i.replace('增加值_当季值(亿元)',") for i in num.index]
plt.barh(label, num)
plt.xlabel(‘产业增加值_当季值(亿元)')
plt.title('2017年第一季度各个行业的生产总值数值条形图’)
plt.show()

#比较2017年第一季度第一二三产业的产业增加值占比情况
num = data.iloc[68, 2:5].values
plt.pie(num,#数据labels=[‘第一产业’,’第二产业’,’第三产业’],# 数据对应的标签autopct=’%.2f%%’，#显示占比比例)
plt.show()

#对于条形图和饼图，可进一步优化数据标注的位置和显示方式。比如条形图中，可将数值标注在条形顶端且居中显示；饼图中，可调整标注位置避免重叠，让占比信息展示更清晰。

#导入相关库
import pandas as pd
import matplotlib.pyplot as plt

#读取相关数据
product=pd.read_excel('./product.xlsx')

#创建自定义的离差标准化函数
def min_max_scale(data):
data=(data-data.min())/(data.max()-data.min())
return data

#调用自定义函数处理数据
product_value1=min_max_scale(product_value1)
product_value2=min_max_scale(product_value2)

#计算综合指标并排序
product['composite_index']=product_value1*0.6+product_value2*0.4
product.sort_values=(by='composite_index',inplace=Ture,ascending=False)
print(product)

#对数据处理流程提出一条优化建议
#可以调用scikit-learn库中preprocessing模块的MinMaxScaler类用于离差标准化的处理。

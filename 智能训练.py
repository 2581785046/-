#加载库
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# 读入数据集
data = np.load('./mnist.npz')
data.files
#数据查看
x_train, y_train, x_test, y_test = \
 data['x_train'], data['y_train'], data['x_test'], data['y_test']
print('训练集样本维度：', x_train.shape)
print('训练集标签维度：', y_train.shape)
print('测试集样本维度：', x_test.shape)
print('测试集标签维度：', y_test.shape)
#数据标准化
x_train, x_test = x_train / 255.0, x_test / 255.0 #除以255归一到0-1
#绘图
plt.figure()
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
plt.imshow(x_train[0],cmap=plt.cm.binary) #默认为彩色，加cmap=plt.cm.binary为黑白
plt.colorbar() #是否显示颜色轴
plt.grid(False)
plt.title('图片里数字是：%d'%y_train[0],size = 15)
plt.show()
#搭建单层RNN
model = tf.keras.models.Sequential()
model.add(layers.RNN(layers.SimpleRNNCell(100), #100个RNN节点
 input_shape=(28, 28))) #输入维度为：28X28
model.add(layers.Dense(10, activation='softmax')) #添加输出层
model.summary() #查看模型图层
#模型编译，参数设置
model.compile(loss='sparse_categorical_crossentropy',
optimizer='adam',
metrics=["accuracy"])
#模型训练
#迭代训练5次，loss不断下降，准确率提高
history = model.fit(x_train, y_train, epochs=5,
 validation_data=(x_test,y_test))
#损失函数变化
import matplotlib.pyplot as plt
plt.plot(history.epoch,history.history.get('loss'),label='loss')
plt.plot(history.epoch,history.history.get('val_loss'),label='val_loss')
plt.legend()
plt.show()
#准确率变化
plt.plot(history.epoch,history.history.get('accuracy'),label='acc')
plt.plot(history.epoch,history.history.get('val_accuracy'),label='val_acc')
plt.legend()
plt.show()
#模型验证结果
model.evaluate(x_test,y_test,verbose=2)
#模型预测
predictions = model.predict(x_test)
print(np.argmax(predictions[0])) #预测第0张图片的结果
print(y_test[0]) #原数据第0张数字
# 绘制图像函数,预测结果展示
i = 4
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(x_test[i], cmap=plt.cm.binary)
plt.xlabel('原图片的数字是：%d'%y_test[i],color='red',size = 15)
# 绘制概率值分布函数
plt.subplot(1,2,2)
plt.xticks(range(10),size = 12)
my_plot = plt.bar(range(10), predictions[i],color='blue')
plt.ylim([0, 1])
plt.show()


# # 模型优化的方法
# 1、调整超参数：尝试不同的学习率、批量大小和迭代次数等超参数，以找到最佳的配置。可以使用交
# 叉验证或网格搜索等技术来系统地搜索最优超参数组合。
# 2、正则化技术：考虑添加正则化技术，如L1正则化或L2正则化，在模型训练过程中引入惩罚项，以减
# 少过拟合现象。
# 3、更复杂的模型结构：尝试增加更多的隐藏层和神经元，以增加模型的表达能力。然而，要小心过度
# 拟合的风险，可能需要进行适当的正则化或使用其他防止过拟合的技术。
# 4、数据增强：通过对训练数据进行增强操作，如随机裁剪、平移、旋转等，来增加训练集的多样性，
# 使模型更具鲁棒性。
# 5、使用预训练模型：考虑使用预训练的模型作为初始权重或进行迁移学习，尤其当数据集较小时，这
# 可以帮助提高模型的性能。
# 6、调整网络层和节点数：尝试增加或减少隐藏层的数量，以及每个隐藏层中的神经元数量。不同的网
# 络结构可能对不同的问题和数据集更有效。
# 7、优化算法：除了常用的随机梯度下降（SGD）之外，还可以尝试其他优化算法，如Adam、
# RMSprop等，以及其不同的参数设置。
# 8、特征工程：对输入数据进行适当的预处理和特征工程，可以帮助提取更有信息量的特征，从而改善
# 模型的性能。

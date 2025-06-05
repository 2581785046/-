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
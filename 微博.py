# 导入相关库
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import re
import jieba
from sklearn.model_selection import train_test_split
from tkinter import _flatten
new_data = pd.read_csv('./nCoV.csv')
print(new_data.shape)
print(new_data.columns)
## 分词
jieba.load_userdict('./newdict.txt')
t = new_data['微博中文内容'].astype(str).apply(lambda x: re.findall('#([^#]+)#', x))
topic_sets = set(_flatten(list(t)))
for i in topic_sets:
 jieba.add_word(i)
data_cut = new_data['微博中文内容'].apply(jieba.lcut)
## 去除停用词
with open('./stopwords.dat', encoding='utf-8') as f:
 stop = f.read().split()
stop = [' ', '\n', '\t', '##', '\ue627', '\u3000', '\ue335', '"'] + stop
data_after = data_cut.apply(lambda x: [i for i in x if i not in stop])
## 构建ID与词典的映射关系
words = set(_flatten(list(data_after)))
word_index = {w:(i+3) for i, w in enumerate(words)}
word_index['_PAD'] = 0
word_index['_START'] = 1
word_index['_UNK'] = 2
word_index['_UNUSE'] = 3
## 构建ID向量
reverse_word_index = dict(zip(word_index.values(), word_index.keys()))
def encode(txt):
 return [word_index.get(i, word_index['_UNK']) for i in txt]
ind = data_after.apply(len) > 5
x = data_after[ind].apply(encode)
## 统一长度：Padding
max_len = 60
x2 = tf.keras.preprocessing.sequence.pad_sequences(
 x, value=word_index['_PAD'], maxlen=max_len, padding='post'
)
## 划分训练集、测试集
y = new_data.loc[x.index, '情感倾向'].map({-1: 0, 0:1, 1:2})
x_train, x_test, y_train, y_test = train_test_split(x2, tf.one_hot(y, 3).numpy(), test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
## 构建模型
vocab_size = len(word_index)
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 32))
model.add(keras.layers.Bidirectional(keras.layers.GRU(64, return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.GRU(64)))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 模型训练
history = model.fit(x_train, y_train, epochs=1, verbose=1)
print('Trainning Complete'.center(50, '='))
# 模型评估
y_pre_prob = model.predict(x_test)
y_pre = np.argmax(y_pre_prob, axis=1)
np.mean(y_pre == np.argmax(y_test, axis=1))

#数据标注化与离散化培训大纲
# 培训课时
# 18课时
# 培训内容
# 数据变换的方法
# 数据标准化的概念
# 数据标准化的方法
# 数据离散化的概念
# 数据离散化的方法
# 独热编码的概念
# 独热编码的方法


# 用户--语音指令\手机app--智能家居系统--家电设备
#
# 每一个步骤的详细操作和可能出现的情况
# 第一步，用户发出语音指令或使用手机App控制智能家居系统：用户的语音指令无法识别：
# 智能家居系统会播放提示音并要求用户重新发出指令；手机App连接失败：智能家居系统会
# 检查网络连接，如果出现问题，会提醒用户检查网络设置。
# 第二步，智能家居系统接收到用户的指令，并根据指令进行相应操作：智能家居系统无法解
# 析指令：系统会播放提示音，并向用户解释无法理解指令的原因；智能家居系统成功解析指
# 令：系统会执行相应的操作，并通过语音或手机App反馈操作结果给用户。
# 第三步，智能家居系统根据指令控制相关的家电设备：家电设备无响应：系统会尝试多次发
# 送指令，如果仍然无法控制设备，系统会向用户报告设备故障并提供相应的解决建议；家电
# 设备成功执行指令：系统会通过语音或手机App向用户确认家电设备已成功执行相应指令。
# （3）该智能家居系统的优势：
# 方便快捷的人机交互方式：用户可以通过语音指令或手机App远程控制家电设备，不需要手
# 动操作开关或前往设备位置。
# 智能化的指令解析与操作执行：系统能够较准确地解析用户的指令，并执行相应操作，提高
# 了用户的使用体验。
# 远程控制功能：用户可以通过手机App在任何地方远程控制家电设备，增加了便利性和灵活
# 性。
# 该智能家居系统的不足：
# 语音指令识别准确度有限：系统对于用户指令的识别准确度可能存在一定的局限性，尤其是
# 在复杂的环境噪声下。
# 网络连接依赖性较强：为了实现远程控制功能，系统要求用户的手机App和智能家居系统保
# 持稳定的网络连接，如果网络出现问题，可能影响控制效果。
# 部分家电设备兼容性差：由于不同品牌、型号的家电设备之间存在兼容性差异，系统可能无
# 法对所有设备进行完全控制。用户需要选择与智能家居系统兼容的设备。



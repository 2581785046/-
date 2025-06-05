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
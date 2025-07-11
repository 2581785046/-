import pandas as pd
import tensorflow as tffrom tensorflow 
import kerasimport numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection 
import train_test_split
# 加载数据
data = np.load('./fashion.npz')
data.files
images, labels = data['images'], data['labels']# 数据查看
print(images.shape)
print(labels.shape)
# 可视化
plt.figure()
plt.imshow(images[1]) # 查看训练集中的其中一幅图
plt.show()
# 数据处理（对图片进行归一化）
images = images / 255.0
# 划分训练集和验证集
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=123)
print('训练集样本维度：', train_images.shape)
print('训练集标签维度：', train_labels.shape)
print('测试集样本维度：', test_images.shape)
print('测试集标签维度：', test_labels.shape)
# 模型搭建
model = tf.keras.Sequential()  # 创建一个Sequential模型
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 添加第一层，展开
model.add(tf.keras.layers.Dense(128, activation='relu'))  # 添加Dense层，可多加几层
model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 输出层
# 模型编译，参数设置
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.summary()  # 查看模型
# 模型训练
history = model.fit(train_images, train_labels, epochs=10)  # 迭代次数为10
# 可视化：loss和accuracy的变化方向
plt.plot(history.epoch, history.history.get('loss'))
plt.plot(history.epoch, history.history.get('accuracy'))
plt.legend(['loss', 'accuracy'])
plt.show()  # 添加显示图表命令
# 模型预测
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))  # 预测第1张图片的类别
print(test_labels[0])  # 数据第1张图片的类别
# 模型验证
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n验证集准确率：', test_acc)


三
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model 
import LinearRegression
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = 'SimHei'  # 修正了原始代码中的拼写错误'sans-scrif'->'sans-serif'
# 加载数据
df = pd.read_csv('/data/data.csv', encoding='gb18030')
print(df.head())  # 查看数据前几行
# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(df['工作时间（小时/周）'], df['病人满意度（评分）'])
plt.xlabel('工作时间（小时/周）')
plt.ylabel('病人满意度（评分）')
plt.title('医生工作时间与病人满意度之间的关系')
plt.grid(True)  # 添加网格线
plt.show()
# 准备数据
X = df['工作时间（小时/周）'].values.reshape(-1, 1)
y = df['病人满意度（评分）'].values
# 训练线性回归模型
regressor = LinearRegression()
regressor.fit(X, y)
# 提取系数和决定系数
coefficient = regressor.coef_[0]
r_squared = regressor.score(X, y)
# 打印结果
print(f"医生的工作时间对病人满意度的影响程度：")
print(f"系数：{coefficient:.3f}")  # 添加了缺失的右括号
print(f"决定系数：{r_squared:.3f}")  # 添加了缺失的右括号
# 计算工作时间增加对满意度的影响
time_increase = 8
satisfaction_change = time_increase * coefficient
print(f"工作时间增加{time_increase}小时/周，预计病人满意度变化: {satisfaction_change:.3f}分")




#语音识别准确性不高的原因及解决方案（精简版）：
#1. 原因：数据质量差 (噪声多、多样性不足)
#2. 方案：收集高质量多样化数据 + 数据预处理 (降噪、增强)
#3. 原因：口音/方言适应性差
#4. 方案：增加口音/方言数据 + 领域适应技术
#5. 原因：语速/语调/发音变化敏感
#6. 方案：增加多样性数据 + 优化模型架构/算法

#评估与改进自然语言理解流程（精简版）：
#1. 准备数据： 收集多样化测试数据（覆盖领域、难度、指令类型）。
#2. 设定指标： 定义关键评估指标（意图、实体、语义解析、上下文理解准确率）。
#3. 人工检验： 进行人工评估（专家模拟用户，打分评价解析结果）。

#关系： 医生工作时间与病人满意度呈正相关。
#回归分析：
#系数 = 0.026 (每增加1小时工作，满意度+0.026分)
#决定系数 R² = 0.747 (工作时间解释了74.7%的满意度变异)
#计算示例：
#工作时间 +8小时 → 满意度 +0.21分 (0.026 * 8)
#工作时间 2000小时 → 预测满意度 59.71分
#应用： 为智慧医疗医生排班和提升病人满意度提供参考。
#局限： 结论基于有限样本，可靠性受限；需更多研究验证

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

raw_dataset = np.loadtxt('watermelon_3a.csv', delimiter=",")  # 加载数据集
sugar_and_density_dataset = raw_dataset[:, 1:3]  # 选取含糖率列与密度列
x = sugar_and_density_dataset

label = raw_dataset[:, 3]  # 选取标记列
y = label

# print(x[y==0, 0])  # array特性

# --------------------------------绘图--------------------------------
fig = plt.figure(1)

plt.title('西瓜数据集')
plt.xlabel('密度')
plt.ylabel('含糖度')
plt.axis([0.2, 0.9, 0, 0.5])
plt.scatter(x[y == 0, 0], x[y == 0, 1], marker = 'o', color = 'k', s=100, label = '坏瓜')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker = 'o', color = 'g', s=100, label = '好瓜')
plt.legend(loc = 'upper right')
plt.show(block = False)
# --------------------------------------------------------------------


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.5, random_state=0)
# train_data        所要划分的样本特征集——x
# train_target      所要划分的样本标记——y
# test_size         样本占比，如果是整数的话就是样本的数量
# random_state      是随机数的种子。
#                   随机数种子其实就是该组随机数的编号。
#                   在需要重复试验的时候，保证得到一组一样的随机数。
#                   比如每次都填1，其他参数一样的情况下得到的随机数组是一样的。
#                   但填0或不填，每次都会不一样。


log_model = LogisticRegression()
# penalty       正则化参数，三种取值：‘l1’, ‘l2’, ‘elasticnet’, 默认=’l2’。
#               如果使用 l1 ，部分接近 0 的参数会被 0 取代
# C             必须为正浮点数。C 越小对损失函数的惩罚越重。
# class_weight  样本权重，可以是一个字典或者 ’balanced’ 字符串，默认为 None。
#               对于二分类模型，可以这样指定权重：class_weight={0:0.9,1:0.1}。
#               当 class_weight=‘balanced’，那么类库会根据训练样本量来计算权重。
#               某种类型样本量越多，则权重越低，样本量越少，则权重越高。
# solver        优化算法选择参数，五种取值：newton-cg, lbfgs, liblinear, sag, saga。默认是 liblinear。
#               五种方式分别是：牛顿法、拟牛顿法、梯度下降法、随机梯度下降、随机梯度下降的优化。
#               liblinear 适用于小数据集，而 sag 和 saga 适用于大数据集，因为速度更快。
#               如果是 l2 正则化，那么除了 saga 都可以选择。
#               但是如果 penalty 是 l1 正则化的话，就只能选择 liblinear 和 saga 了。


log_model.fit(x_train, y_train)  # 装载训练样本和标记向量

y_predict = log_model.predict(x_test)  # 用测试集预测样本

cm = metrics.confusion_matrix(y_test, y_predict)  # 计算混淆矩阵
#  TP FN
#  FP TN
# 查准率  TP / (TP + FP)
# 查全率  TP / (TP + FN)

cr = metrics.classification_report(y_test, y_predict)  # 全面评估预测结果
#              precision    recall  f1-score   support
#                  查准率     查全率      F1值       总数
#         0.0       0.80      0.80      0.80         5
#         1.0       0.75      0.75      0.75         4
#
#    accuracy                           0.78         9     精度
#   macro avg       0.78      0.78      0.78         9     算术平均值 (0.8+0.75)/2
#weighted avg       0.78      0.78      0.78         9     加权平均值 (0.8*5+0.75*4)/9


precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_predict)  # P-R曲线

# --------------------------------绘图--------------------------------
fig_2 = plt.figure(2)

plt.title('P-R曲线')
plt.xlabel('查全率')
plt.ylabel('查准率')
plt.plot(recall, precision, c = 'k')
plt.show(block = False)
# --------------------------------------------------------------------

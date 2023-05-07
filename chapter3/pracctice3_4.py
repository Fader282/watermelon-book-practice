import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression  # 对率回归
from sklearn.model_selection import LeaveOneOut  # 留一法
from sklearn import metrics
from sklearn.model_selection import cross_val_predict  # k折交叉检验

iris = np.loadtxt('iris.data', delimiter=",", dtype='str')  # 加载数据集
x_ = iris[:150,0:4]
y_ = iris[:150,4]
for i in range(150):  # 三种花，改为0，1，2
    if y_[i] == 'Iris-setosa':
        y_[i] = 0
    elif y_[i] == 'Iris-versicolor':
        y_[i] = 1
    elif y_[i] == 'Iris-virginica':
        y_[i] = 2

# 全部取出来
x = np.array(x_[:150], dtype='float')
y = np.array(y_[:150], dtype='float')

# ------------------ 图形化显示数据集 ------------------
sns.set(style="white", color_codes=True)
fig = plt.figure(1)
iris = sns.load_dataset(name = 'iris', data_home='seaborn-data', cache=True)
s1 = sns.pairplot(iris, hue='species')
plt.show()
# ---------------------------------------------------

# -------------------- 十折交叉验证法 ------------------
log_model = LogisticRegression(max_iter=3000)  # 使用对率回归，增加最大迭代次数
y_pred_cvp = cross_val_predict(log_model, x, y, cv=10)
# 得到经过 K = 10 折交叉验证计算得到的每个训练验证的输出预测

acc_cvp = metrics.accuracy_score(y, y_pred_cvp)  # 获得正确率
# ----------------------------------------------------


# ---------------------- 留一法 -----------------------
loo = LeaveOneOut()
accuracy = 0  # 验证正确的数量
for train, test in loo.split(x):
    log_model.fit(x[train], y[train])  # 装填训练集
    y_pred_loo = log_model.predict(x[test])  # 测试集，但是只有一个数据
    if y_pred_loo == y[test]:
        accuracy += 1
acc_loo = accuracy / np.shape(x)[0]  # 正确率
# ----------------------------------------------------
# [ 0  1  2  3  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
#  25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
#  49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72
#  73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96
#  97 98 99] [4]  train的值与test的值

print(f'<-----鸢尾花数据集----->\n十折交叉验证法的正确率：{acc_cvp * 100}%\n留一法的正确率：{acc_loo * 100}%\n')



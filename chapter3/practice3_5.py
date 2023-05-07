import numpy as np
import matplotlib.pyplot as plt

REMOVE_15 = True  # 是否删掉第15行（第14个样本）
# 将西瓜数据集中的bad类离群点15删去后数据集的线性可分性大大提高。


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 导入西瓜数据集
data_file = open('watermelon_3a.csv')
dataset = np.loadtxt(data_file, delimiter=",")
# ndarray矩阵↑

# X 提取C2, C3列，Y提取C4列
X = dataset[:, 1:3]
y = dataset[:, 3]

if REMOVE_15:
    X = np.delete(X, 14, axis=0)
    y = np.delete(y, 14, axis=0)

# 图形化数据显示
def show_data():
    f1 = plt.figure(1)
    plt.title('西瓜3.0α数据集')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    # 描点用这个函数↓
    plt.scatter(X[y == 0,0], X[y == 0,1], marker = '_', color = 'k', s=100, label = '坏瓜')
    plt.scatter(X[y == 1,0], X[y == 1,1], marker = '+', color = 'g', s=100, label = '好瓜')
    plt.legend(loc = 'upper left')
    plt.show()


# u[0]密度正反两类均值，u[1]含糖率正反两类均值，得均值向量μ0，μ1
u = []
for i in range(2):
    u.append(np.mean(X[y==i], axis=0))
print(u)


m,n = np.shape(X)
Sw = np.zeros((n,n))
# 申明空矩阵用这个函数↑

# 求均值向量
u0 = u[0].reshape(n, 1)
u1 = u[1].reshape(n, 1)
print(f'0类均值向量{u0}, 1类均值向量{u1}')

# 对每个样本都操作一次，每个样本里又有两个属性，与各属性的均值做差后求点积，求两个类的协方差矩阵
E0 = np.zeros((n,n))
E1 = np.zeros((n,n))
for i in range(m):
    x_tmp = X[i].reshape(n,1)
    if y[i] == 0:  # 0类的协方差
        E0 += np.dot( x_tmp - u0, (x_tmp - u0).T )
    if y[i] == 1:  # 1类的协方差
        E1 += np.dot( x_tmp - u1, (x_tmp - u1).T )
print(f'Σ0={E0}, Σ1={E1}')

Sw = E0 + E1  # 类内散度矩阵Sω (3.33)

# 可以按书上使用奇异值分解np.linalg.svd()的方法求逆矩阵，但是可以直接使用np.linalg.inv()求逆矩阵
# 求逆矩阵
Sw_inv = np.linalg.inv(Sw)

# 得到投影方向ω (3.39)
w = np.dot( Sw_inv, u0 - u1 )
print(w)


# *绘制投影查看类簇情况
# https://github.com/PnYuan/Machine-Learning_ZhouZhihua/tree/master/ch3_linear_model/3.5_LDA
def show_DLA_result():
    f4 = plt.figure(2)
    plt.xlim( -0.2, 1 )
    plt.ylim( -0.5, 0.7 )

    p0_x0 = -X[:, 0].max()
    p0_x1 = ( w[1,0] / w[0,0] ) * p0_x0
    p1_x0 = X[:, 0].max()
    p1_x1 = ( w[1,0] / w[0,0] ) * p1_x0

    plt.title('西瓜数据集3.0a，线性判别分析')
    plt.xlabel('密度')
    plt.ylabel('含糖率')
    plt.scatter(X[y == 0,0], X[y == 0,1], marker = '_', color = 'k', s=30, label = '坏瓜')
    plt.scatter(X[y == 1,0], X[y == 1,1], marker = '+', color = 'g', s=30, label = '好瓜')
    plt.legend(loc = 'upper right')

    plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])

    def GetProjectivePoint_2D(point, line):
        a = point[0]
        b = point[1]
        k = line[0]
        t = line[1]

        if   k == 0:
            return [a, t]
        elif k == np.inf:
            return [0, b]
        x = (a+k*b-k*t) / (k*k+1)
        y = k*x + t
        return [x, y]

    m,n = np.shape(X)
    for i in range(m):
        x_p = GetProjectivePoint_2D( [X[i,0], X[i,1]], [w[1,0] / w[0,0] , 0] )
        if y[i] == 0:
            plt.plot(x_p[0], x_p[1], 'ko', markersize = 5)
        if y[i] == 1:
            plt.plot(x_p[0], x_p[1], 'go', markersize = 5)
        plt.plot([ x_p[0], X[i,0]], [x_p[1], X[i,1] ], 'c--', linewidth = 0.3)

    plt.show()


if __name__ == '__main__':
    # show_data()
    show_DLA_result()
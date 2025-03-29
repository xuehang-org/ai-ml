---
title: 常用算法 K 近邻算法
---


# 常用算法 K 近邻算法

K 近邻 (KNN) 是一种简单但功能强大的监督学习算法，可用于分类和回归问题。它的核心思想是：**一个样本的类别由其最近的 K 个邻居的类别决定。**  简单来说，就是“近朱者赤，近墨者黑”。

## 算法原理

1.  **确定 K 值：** 选择一个合适的 K 值，即选择多少个邻居来做决策。 K 值的选择对结果影响很大。
2.  **计算距离：** 计算目标样本与所有训练样本之间的距离。 常用的距离度量方法包括：
    *   **欧氏距离：**  最常用的距离公式，就是两点之间的直线距离。
    *   **曼哈顿距离：**  只能沿着坐标轴方向移动的距离，也称为“城市街区距离”。
    *   **闵可夫斯基距离：**  是欧氏距离和曼哈顿距离的推广。
3.  **寻找邻居：**  选择距离目标样本最近的 K 个训练样本，作为它的 K 个邻居。
4.  **做出预测：**
    *   **分类问题：**  根据 K 个邻居中出现次数最多的类别，将目标样本归为该类。
    *   **回归问题：**  将 K 个邻居的平均值作为目标样本的预测值。

## KNN 的优缺点

**优点：**

*   **简单易懂：**  原理简单，容易实现。
*   **无需训练：**  KNN 是一种“懒惰学习”算法，不需要显式的训练过程，直接利用已有的数据进行预测。
*   **适用性广：**  可用于分类和回归问题。
*   **对异常值不敏感：**  由于采用多数投票或平均值方法，单个异常值的影响较小。

**缺点：**

*   **计算量大：**  预测时需要计算目标样本与所有训练样本的距离，当数据量大时，计算开销很大。
*   **需要大量存储空间：**  需要存储所有的训练样本。
*   **K 值选择敏感：**  K 值的选择对结果影响很大，需要通过交叉验证等方法选择合适的 K 值。
*   **不擅长处理高维数据：**  在高维空间中，距离的意义变得模糊，容易导致“维度灾难”。
*   **数据不平衡问题：**  当样本类别不平衡时，容易将目标样本分到样本数量较多的类别。

## Scikit-learn 中的 KNN

Scikit-learn 提供了 `KNeighborsClassifier` 类用于分类问题，`KNeighborsRegressor` 类用于回归问题。

### KNN 分类

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 1. 加载数据集
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # 为了方便可视化，只使用前两个特征

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建 KNN 分类器
knn = KNeighborsClassifier(n_neighbors=5)  # 设置 K 值为 5

# 4. 训练模型
knn.fit(X_train, y_train)

# 5. 预测
y_pred = knn.predict(X_test)

# 6. 评估模型
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy}")

# 7. 可视化决策边界
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

h = .02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# 将结果放入一个颜色图中
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# 绘制训练点
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (5))
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend() #添加图例，避免显示中文
plt.show()
```

![](/23.png)
*Fig.23*

**代码解释：**

*   `load_iris()`:  加载鸢尾花数据集。
*   `train_test_split()`:  将数据集划分为训练集和测试集。
*   `KNeighborsClassifier(n_neighbors=5)`:  创建一个 KNN 分类器，设置 K 值为 5。
*   `fit(X_train, y_train)`:  使用训练数据训练模型。
*   `predict(X_test)`:  使用训练好的模型对测试数据进行预测。
*   `np.mean(y_pred == y_test)`:  计算预测的准确率。
*   代码中使用了 `matplotlib` 库进行可视化，展示了 KNN 分类器的决策边界。

### KNN 回归

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成一些示例数据
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8)) # 添加一些噪声

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建 KNN 回归器
knn = KNeighborsRegressor(n_neighbors=5) # 设置 K 值为 5

# 4. 训练模型
knn.fit(X_train, y_train)

# 5. 预测
y_pred = knn.predict(X_test)

# 6. 评估模型 (例如，均方误差)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# 7. 可视化结果
plt.figure()
plt.scatter(X_train, y_train, c='darkorange', label='Training data')
plt.scatter(X_test, y_test, c='lightgreen', label='Testing data')
plt.plot(X_test, y_pred, c='red', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('KNeighborsRegressor (k = 5)')
plt.legend() #添加图例，避免显示中文
plt.show()
```

![](/24.png)
*Fig.24*

**代码解释：**

*   `KNeighborsRegressor(n_neighbors=5)`: 创建一个 KNN 回归器，设置 K 值为 5。
*   `np.mean((y_pred - y_test) ** 2)`:  计算均方误差 (Mean Squared Error)，用于评估回归模型的性能。
*   代码使用 `matplotlib` 库进行可视化，展示了 KNN 回归的预测结果。

## 如何选择 K 值

K 值的选择对 KNN 算法的性能至关重要。

*   **K 值过小：**  模型容易受到噪声的影响，导致过拟合。
*   **K 值过大：**  模型会变得过于简单，可能欠拟合，并且计算量也会增大。

**常用的 K 值选择方法：**

*   **交叉验证：**  将数据集划分为多个子集，轮流选择其中一个子集作为验证集，其余子集作为训练集。尝试不同的 K 值，选择在验证集上性能最好的 K 值。
*   **经验法则：**  通常选择 K 为奇数，以避免在二分类问题中出现平票的情况。

## 距离度量方法的选择

不同的距离度量方法会对 KNN 算法的性能产生影响。

*   **欧氏距离：**  最常用的距离度量方法，适用于连续型数据。
*   **曼哈顿距离：**  适用于维度之间不相关的情况。
*   **其他距离度量方法：**  例如，余弦相似度、切比雪夫距离等，可以根据具体情况选择。

## 数据预处理

KNN 算法对数据的scale比较敏感。 因此，在使用 KNN 算法之前，通常需要进行数据预处理，例如：

*   **标准化 (Standardization):** 将数据缩放到均值为 0，方差为 1 的范围。
*   **归一化 (Normalization):** 将数据缩放到 0 到 1 的范围。

## 总结

K 近邻算法是一种简单而有效的机器学习算法。 掌握 KNN 的原理、优缺点、以及如何在 Scikit-learn 中使用 KNN，可以帮助你解决许多实际问题。

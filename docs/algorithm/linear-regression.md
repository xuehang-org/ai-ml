---
title: 常用算法 线性回归
---

# 常用算法 线性回归

线性回归是一种基础且强大的机器学习算法，用于建立自变量（特征）与因变量之间的线性关系模型。简单来说，就是用一条直线（或者更高维的超平面）来拟合数据，从而预测未知的数据点。

## 概念

*   **自变量 (Independent Variable / Feature):**  影响因变量的因素，通常用 `X` 表示。
*   **因变量 (Dependent Variable / Target):**  我们想要预测的变量，通常用 `y` 表示。
*   **线性关系:**  自变量和因变量之间存在可以用一条直线（或超平面）描述的关系。
*   **模型:**  `y = wX + b`，其中 `w` 是权重（斜率），`b` 是截距（bias）。 线性回归的目标就是找到最佳的 `w` 和 `b`，使得模型能够最好地拟合数据。

## 算法原理

线性回归试图找到一条最佳的直线（或超平面），使得所有数据点到这条直线的距离（误差）最小。 常用的误差衡量标准是**最小二乘法**，即所有数据点的预测值与真实值之差的平方和最小。

## scikit-learn 实现

scikit-learn 提供了 `LinearRegression` 类来实现线性回归。

### 引入库

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
```

### 数据准备

首先，我们需要准备一些数据。这里我们生成一些简单的线性数据：

```python
X = np.array([[1], [2], [3], [4], [5]])  # 自变量
y = np.array([2, 4, 5, 4, 5])  # 因变量
```

### 创建模型并训练

```python
model = LinearRegression()  # 创建线性回归模型
model.fit(X, y)  # 使用数据训练模型
```

### 预测

```python
X_new = np.array([[6]])  # 准备要预测的新数据
y_pred = model.predict(X_new)  # 使用模型进行预测
print(f"预测值: {y_pred[0]:.2f}")
```

### 模型参数

训练完成后，我们可以查看模型的权重（斜率）和截距：

```python
print(f"权重 (w): {model.coef_[0]:.2f}")
print(f"截距 (b): {model.intercept_:.2f}")
```

### 可视化

将数据和模型可视化可以更直观地理解线性回归的效果。

```python
plt.scatter(X, y, label="Data")  # 绘制数据点
plt.plot(X, model.predict(X), color='red', label="Linear Regression")  # 绘制线性回归线
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
```

![](/14.png)
*Fig.14*

### 完整代码示例

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据准备
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# 2. 创建模型并训练
model = LinearRegression()
model.fit(X, y)

# 3. 预测
X_new = np.array([[6]])
y_pred = model.predict(X_new)
print(f"预测值: {y_pred[0]:.2f}")

# 4. 模型参数
print(f"权重 (w): {model.coef_[0]:.2f}")
print(f"截距 (b): {model.intercept_:.2f}")

# 5. 可视化
plt.scatter(X, y, label="Data")
plt.plot(X, model.predict(X), color='red', label="Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
```

## 更多示例

### 多元线性回归

当有多个自变量时，就变成了多元线性回归。

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 数据准备 (多个自变量)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])  # 两个自变量
y = np.array([3, 5, 7, 9, 11])

# 2. 创建模型并训练
model = LinearRegression()
model.fit(X, y)

# 3. 预测
X_new = np.array([[6, 7]])
y_pred = model.predict(X_new)
print(f"预测值: {y_pred[0]:.2f}")

# 4. 模型参数
print(f"权重 (w): {model.coef_}")
print(f"截距 (b): {model.intercept_:.2f}")

# 5. 可视化 (仅用于两个自变量的情况，方便展示)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y, label="Data")

# 创建一个网格用于绘制预测平面
x0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
x1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
x0, x1 = np.meshgrid(x0, x1)
X_grid = np.column_stack((x0.ravel(), x1.ravel()))
y_pred_grid = model.predict(X_grid).reshape(x0.shape)

ax.plot_surface(x0, x1, y_pred_grid, alpha=0.5, color='red', label="Linear Regression")

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")
ax.set_title("Multiple Linear Regression Example")
ax.legend()
plt.show()
```

![](/15.png)
*Fig.15*

### 更多数据

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate more data
np.random.seed(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)  # More data points
y = 2 * X + 1 + np.random.normal(0, 2, size=(50, 1))  # Add some noise

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_test)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label='Training Data')
plt.plot(X_test, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with More Data')
plt.legend()
plt.show()
```

![](/16.png)
*Fig.16*

## 优点

*   简单易懂，容易实现。
*   计算速度快。
*   可以用于预测和解释。

## 缺点

*   只能处理线性关系。
*   对异常值敏感。
*   可能出现过拟合。

## 使用场景

*   房价预测
*   销售额预测
*   股票价格预测 (短期)
*   ...任何存在线性关系的场景

希望这个文档能够帮助你理解和使用 scikit-learn 的线性回归算法！

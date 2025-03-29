---
title: 基本概念 欠拟合
---

# 基本概念 欠拟合

## 什么是欠拟合？

想象一下，你想用一个模型来预测房价。

*   **理想情况：** 你找到一个模型，它能很好地根据房子的大小、位置等因素预测房价。
*   **欠拟合：** 你的模型**过于简单**，根本无法捕捉到房价和各种因素之间的复杂关系。就好比你用一条直线来拟合房价数据，但实际上房价和大小之间是非线性的关系。

**简单来说：** 欠拟合就是模型没学到足够的东西，导致它在训练数据和新数据上的表现都不好。

## 欠拟合的特征

*   **训练误差大：** 模型在训练集上的预测效果就很差。
*   **泛化能力差：** 模型在新数据（测试集）上的表现也很差。
*   **模型过于简单：**  模型可能使用了过少的特征，或者选择了过于简单的算法。

## 欠拟合的原因

*   **特征不足：**  模型没有足够的特征来学习数据中的复杂关系。
*   **模型复杂度过低：**  模型本身过于简单，无法表达数据的内在结构。
*   **正则化过度：**  过强的正则化约束了模型的学习能力 (这个我们后面会讲)。

## 如何识别欠拟合？

1.  **观察训练误差：** 如果训练误差很高，那很可能就是欠拟合。
2.  **观察验证误差：**  如果验证误差和训练误差都很高，并且两者接近，那也可能是欠拟合。
3.  **学习曲线：** 绘制学习曲线，观察训练集和验证集的误差变化。如果两条曲线都收敛到一个较高的误差水平，那很可能就是欠拟合。

## 如何解决欠拟合？

1.  **增加特征：**  尝试增加更多的相关特征，让模型有更多信息可以学习。
2.  **提高模型复杂度：**  选择更复杂的模型，例如从线性回归换成多项式回归，或者使用更复杂的算法，如神经网络。
3.  **减少正则化：**  如果使用了正则化，可以适当减小正则化的强度，让模型更自由地学习。
4.  **特征工程：** 尝试创造新的特征，例如组合现有特征、进行特征变换等。

## 欠拟合示例

### 1.线性模型拟合非线性数据

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 创建一些非线性数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.3, 100)

# 使用线性回归模型
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)  # 注意将X转换为列向量

# 预测
y_pred = model.predict(X.reshape(-1, 1))

# 绘制结果
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Underfitting Example: Linear Regression on Non-linear Data')
plt.legend()
plt.show()
```

![](/9.png)
*Fig.9*

**分析：**

*   我们用线性回归模型来拟合一个正弦曲线。
*   线性模型无法捕捉到正弦曲线的非线性特征，导致拟合效果很差，这就是欠拟合。

### 2. 决策树模型深度限制

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 创建一些数据
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.3, 100)

# 创建一个深度为2的决策树模型
model = DecisionTreeRegressor(max_depth=2)
model.fit(X.reshape(-1, 1), y)

# 预测
y_pred = model.predict(X.reshape(-1, 1))

# 绘制结果
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Decision Tree (max_depth=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Underfitting Example: Decision Tree with Limited Depth')
plt.legend()
plt.show()
```
![](/10.png)
*Fig.10*

**分析：**

*   我们使用了一个最大深度为2的决策树模型。
*   由于树的深度限制，模型无法充分学习数据的复杂性，导致欠拟合。

### 3. 数据量不足

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 创建少量数据
np.random.seed(0)
X = np.linspace(0, 10, 10)  # 只有10个数据点
y = 2 * X + 1 + np.random.normal(0, 2, 10)

# 使用线性回归模型
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)

# 预测
y_pred = model.predict(np.linspace(0, 10, 100).reshape(-1, 1))

# 绘制结果
plt.scatter(X, y, label='Actual Data')
plt.plot(np.linspace(0, 10, 100), y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Underfitting Example: Insufficient Data')
plt.legend()
plt.show()
```
![](/11.png)
*Fig.11*

**分析：**

*   我们只使用了少量的数据点来训练模型。
*   由于数据量不足，模型无法学习到数据的真实分布，导致欠拟合。

## 总结

欠拟合是指模型学习不足，无法捕捉到数据中的复杂关系。 解决欠拟合的方法包括增加特征、提高模型复杂度、减少正则化等。理解欠拟合的概念，可以帮助我们更好地选择和调整模型，从而获得更好的预测效果。

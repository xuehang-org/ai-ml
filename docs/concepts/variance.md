---
title: 基本概念 方差
---

# 基本概念 方差

## 什么是方差？

在机器学习中，**方差** 用于衡量模型预测结果的离散程度，即预测值偏离真实值的程度。简单来说，方差越大，模型的预测结果越分散；方差越小，模型的预测结果越集中。

*   **高方差 (High Variance):**  模型过度拟合训练数据，对训练数据中的噪声非常敏感。这意味着模型在训练集上表现很好，但在测试集上表现很差。
*   **低方差 (Low Variance):** 模型欠拟合训练数据，无法捕捉到数据中的复杂关系。这意味着模型在训练集和测试集上的表现都不好。

## 方差的数学定义

方差的计算公式如下：

$$
\text{Variance} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}
$$

其中：

*   $x_i$  是单个数据点。
*   $\bar{x}$  是所有数据点的平均值。
*   $n$  是数据点的总数。

注意，这里我们使用了  `n-1`  而不是  `n`，这是因为在统计学中，使用  `n-1`  可以得到方差的无偏估计。

## 如何理解方差？

可以把方差想象成模型预测的稳定性。

*   **高方差:** 就像一个不稳定的射手，每次射击都偏离靶心很远，而且每次偏离的方向都不一样。
*   **低方差:** 就像一个稳定的射手，每次射击都集中在靶心附近，但可能离靶心有点偏差。

## Python 示例

下面我们用 Python 和 scikit-learn 来演示方差的概念。

### 1. 生成模拟数据

首先，我们生成一些模拟数据，用于训练和测试模型。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 生成模拟数据
np.random.seed(0)
X = np.linspace(0, 1, 100)
y = np.cos(2 * np.pi * X) + np.random.normal(0, 0.2, 100)

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 将数据转换为列向量
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]
```

### 2. 创建不同复杂度的模型

我们创建几个不同复杂度的多项式回归模型，来模拟高方差和低方差的情况。

```python
# 创建不同复杂度的多项式回归模型
degrees = [1, 5, 15]
models = []
for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)
    models.append(model)
```

### 3. 评估模型

我们使用均方误差 (Mean Squared Error, MSE) 来评估模型的性能。MSE 越小，模型的性能越好。同时，我们绘制模型的预测结果，以便更直观地理解方差。

```python
# 评估模型
mse_train = []
mse_test = []
for model in models:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))

# 绘制结果
plt.figure(figsize=(12, 6))
plt.suptitle('Polynomial Regression with Different Degrees', fontsize=16)

for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, label='Training Data', color='blue')
    plt.scatter(X_test, y_test, label='Test Data', color='green')
    X_plot = np.linspace(0, 1, 100)[:, np.newaxis]
    y_plot_pred = models[i].predict(X_plot)
    plt.plot(X_plot, y_plot_pred, label=f'Degree {degree} Model', color='red')
    plt.title(f'Degree = {degree}\nMSE Train = {mse_train[i]:.4f}, MSE Test = {mse_test[i]:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
```

![](/13.png)
*Fig.13*

### 4. 分析结果

运行上面的代码，你会看到三个图，分别对应 degree=1, 5, 15 的多项式回归模型。

*   **Degree = 1:** 模型过于简单，无法很好地拟合数据，属于欠拟合 (Underfitting)，方差较低，但偏差较高。
*   **Degree = 5:** 模型复杂度适中，可以较好地拟合数据，泛化能力较好。
*   **Degree = 15:** 模型过于复杂，过度拟合了训练数据，属于过拟合 (Overfitting)，方差较高，但在训练集上偏差较低，在测试集上偏差较高。

从 MSE 的角度来看，Degree=1 的模型在训练集和测试集上的 MSE 都很高，Degree=5 的模型在训练集和测试集上的 MSE 都比较低，Degree=15 的模型在训练集上的 MSE 很低，但在测试集上的 MSE 很高。

## 如何降低方差？

降低方差的一些常用方法包括：

*   **增加训练数据:** 更多的数据可以帮助模型更好地学习数据的真实分布。
*   **简化模型:** 减少模型的复杂度，例如减少多项式回归的 degree，或者使用正则化方法。
*   **正则化:** 通过在损失函数中添加惩罚项，限制模型的复杂度，例如 L1 正则化和 L2 正则化。
*   **集成学习:** 使用多个模型的组合来提高模型的泛化能力，例如 Bagging 和 Random Forest。

## 总结

方差是机器学习中一个重要的概念，它衡量了模型预测结果的离散程度。理解方差的概念，可以帮助我们更好地选择和优化模型，避免过拟合和欠拟合的问题，提高模型的泛化能力。


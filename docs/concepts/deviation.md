---
title: 基本概念 偏差
---

# 基本概念 偏差

在机器学习中，**偏差 (Bias)** 是指模型预测值与真实值之间的**平均差异**。简单来说，偏差衡量了模型在所有可能的训练数据集上预测结果的期望值与真实值之间的差距。

## 偏差的含义

-   **高偏差 (High Bias)**：模型过于简化，无法捕捉数据中的复杂关系。这通常发生在模型过于简单，例如使用线性模型拟合非线性数据时。高偏差模型容易**欠拟合 (Underfitting)**。
-   **低偏差 (Low Bias)**：模型能够很好地拟合训练数据，捕捉数据中的复杂关系。但如果模型过于复杂，可能会导致**过拟合 (Overfitting)**。

可以这样理解：

-   **偏差大**：模型瞄准的目标偏离中心，射击结果整体偏离靶心。
-   **偏差小**：模型瞄准的目标接近中心，射击结果整体接近靶心。

## 偏差的来源

偏差通常来源于以下几个方面：

1.  **模型假设 (Model Assumptions)**：模型本身对数据分布做出了错误的假设。例如，假设数据是线性关系，但实际上是非线性的。
2.  **特征选择 (Feature Selection)**：选择的特征不足以表达数据的真实关系，导致模型无法学习到有效的模式。
3.  **算法限制 (Algorithm Limitations)**：算法本身的限制导致无法学习到数据的全部信息。

## 偏差与方差

偏差通常与**方差 (Variance)** 一起讨论。

-   **方差**衡量的是模型在不同训练数据集上的预测结果的**离散程度**。高方差表示模型对训练数据中的微小变化非常敏感，容易过拟合。

理想情况下，我们希望模型具有**低偏差和低方差**，但通常需要在两者之间进行权衡。这种权衡被称为**偏差-方差权衡 (Bias-Variance Tradeoff)**。

## 如何判断偏差高低？

1.  **观察训练误差和测试误差**：
    -   如果训练误差和测试误差都很高，说明模型可能存在高偏差，需要考虑更换更复杂的模型或增加特征。
    -   如果训练误差很低，但测试误差很高，说明模型可能存在高方差，需要考虑降低模型复杂度或增加数据量。

2.  **学习曲线 (Learning Curve)**：
    -   学习曲线可以帮助我们判断模型是否存在高偏差或高方差问题。
    -   如果训练误差和交叉验证误差收敛到较高的值，说明模型存在高偏差。

## 降低偏差的方法

1.  **选择更复杂的模型**：例如，从线性模型转换为多项式模型或非线性模型。
2.  **增加特征**：增加更多的特征可以帮助模型学习到数据中更丰富的信息。
3.  **使用更复杂的算法**：例如，使用集成学习算法 (如 Gradient Boosting) 代替简单的决策树。

## 示例

下面我们通过一个简单的例子来说明偏差的概念。假设我们有一组非线性数据，我们分别使用线性模型和多项式模型进行拟合。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 生成非线性数据
np.random.seed(0)
X = np.linspace(-5, 5, 100)
y = 0.5 * X**3 - 2 * X**2 + X + 3 + np.random.normal(0, 10, 100)
X = X.reshape(-1, 1)

# 使用线性模型拟合
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)

# 使用多项式模型拟合
poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X, y)
y_poly_pred = poly_model.predict(X)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data')
plt.plot(X, y_linear_pred, color='red', label='Linear Regression')
plt.plot(X, y_poly_pred, color='green', label='Polynomial Regression (degree=3)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Bias Example')
plt.legend()
plt.show()
```

![](/12.png)
*Fig.12*

在这个例子中，线性模型的偏差较高，因为它无法很好地拟合非线性数据。而多项式模型可以更好地拟合数据，偏差较低。

## 总结

偏差是机器学习中一个重要的概念，它衡量了模型预测值与真实值之间的平均差异。理解偏差的来源和影响，可以帮助我们选择合适的模型和算法，提高模型的泛化能力。在实际应用中，我们需要权衡偏差和方差，找到一个平衡点，使模型在训练数据和测试数据上都能表现良好。

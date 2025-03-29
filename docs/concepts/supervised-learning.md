---
title: 基本概念 监督学习
---

# 基本概念 监督学习

监督学习是机器学习中最常见和应用最广泛的类型之一。简单来说，监督学习就像是有一个“老师” (即已标记的训练数据) 来指导模型学习。

## 1. 什么是监督学习？

在监督学习中，我们使用包含**输入特征**（features，也称为自变量）和**目标变量**（target variable，也称为因变量或标签）的已标记数据来训练模型。模型的目标是学习输入特征和目标变量之间的映射关系，以便能够对新的、未见过的数据进行预测。

*   **输入特征 (Features)**: 描述数据的属性。例如，预测房价时，面积、卧室数量、地理位置等都是特征。
*   **目标变量 (Target Variable)**: 我们想要预测的结果。例如，房价本身。
*   **已标记数据 (Labeled Data)**: 包含输入特征和对应的目标变量的数据。

### 1.1 监督学习的流程

1.  **数据准备**: 收集和清洗已标记的数据。
2.  **模型选择**: 选择适合任务的监督学习算法（例如，线性回归、支持向量机、决策树等）。
3.  **模型训练**: 使用已标记的数据训练模型，使其学习输入特征和目标变量之间的关系。
4.  **模型评估**: 使用测试数据评估模型的性能。
5.  **模型部署**: 将训练好的模型部署到实际应用中，对新的数据进行预测。

## 2. 监督学习的类型

监督学习主要分为两类：

*   **回归 (Regression)**: 目标变量是连续值。例如，预测房价、股票价格等。
*   **分类 (Classification)**: 目标变量是离散值。例如，判断邮件是否是垃圾邮件、识别图像中的物体等。

### 2.1 回归 (Regression)

回归任务的目标是预测一个连续的数值。

**常用算法：**

*   **线性回归 (Linear Regression)**: 假设输入特征和目标变量之间存在线性关系。
*   **岭回归 (Ridge Regression)**: 线性回归的正则化版本，用于防止过拟合。
*   **支持向量回归 (Support Vector Regression, SVR)**: 使用支持向量机的回归版本。
*   **决策树回归 (Decision Tree Regression)**: 使用决策树进行回归预测。
*   **随机森林回归 (Random Forest Regression)**: 使用多个决策树的集成进行回归预测。

**示例：线性回归**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[1], [2], [3], [4], [5]])  # 输入特征
y = np.array([2, 4, 5, 4, 5])  # 目标变量

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X, y)

# 预测
X_new = np.array([[6]])
y_pred = model.predict(X_new)
print(f"预测值: {y_pred[0]:.2f}")

# 可视化
plt.scatter(X, y, label="Actual values")
plt.plot(X, model.predict(X), color='red', label="Regression line")
plt.scatter(X_new, y_pred, color='green', label="Predicted value")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Example")
plt.show()
```


![](/1.png)

*Fig.1*

### 2.2 分类 (Classification)

分类任务的目标是将数据划分到不同的类别中。

**常用算法：**

*   **逻辑回归 (Logistic Regression)**: 虽然名字叫回归，但实际上是一种分类算法。
*   **支持向量机 (Support Vector Machine, SVM)**: 寻找最佳超平面来分隔不同类别的数据。
*   **决策树 (Decision Tree)**: 使用树状结构进行分类。
*   **随机森林 (Random Forest)**: 使用多个决策树的集成进行分类。
*   **K近邻 (K-Nearest Neighbors, KNN)**: 根据最近的K个邻居的类别来决定新数据的类别。

**示例：逻辑回归**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 准备数据
X = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 5], [6, 4]])  # 输入特征
y = np.array([0, 0, 0, 1, 1, 1])  # 目标变量 (0 或 1)

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.2f}")

# 可视化
# 注意：由于特征是二维的，可视化需要降维或者选择两个特征进行展示
# 这里简单地打印出预测结果
print("预测结果:", y_pred)
# 准确率: 0.50
# 预测结果: [0 1]
```

## 3. 监督学习的挑战

*   **过拟合 (Overfitting)**: 模型在训练数据上表现很好，但在测试数据上表现很差。
*   **欠拟合 (Underfitting)**: 模型无法很好地拟合训练数据，导致在训练和测试数据上表现都不好。
*   **数据质量**: 数据的缺失、噪声、不平衡等问题会影响模型的性能。

## 4. 总结

监督学习是机器学习中非常重要的一部分，通过已标记的数据训练模型，使其能够对新的数据进行预测。选择合适的算法和处理数据质量问题是构建高性能监督学习模型的关键。

希望这篇文档能够帮助你更好地理解监督学习！

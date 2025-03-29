---
title: 常用算法 逻辑回归
---


# 常用算法 逻辑回归

逻辑回归是一种广泛使用的分类算法，尤其擅长处理二元分类问题。虽然名字里有“回归”，但它实际上是一种分类算法。

## 核心概念

逻辑回归的核心思想是：

1.  **线性回归的底子**：先用线性回归的思路，对特征进行加权求和。
2.  **Sigmoid 函数**：将线性回归的结果，套入一个 Sigmoid 函数中，将输出值压缩到 0 和 1 之间，代表概率。
3.  **概率输出**：输出值越接近 1，代表属于正类的概率越高；反之，越接近 0，代表属于负类的概率越高。
4.  **决策边界**：设定一个阈值（通常是 0.5），当概率大于阈值时，判定为正类，否则为负类。

### Sigmoid 函数

Sigmoid 函数的公式如下：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

其中，$z$ 是线性回归的输出，即 $z = w^T x + b$。

Sigmoid 函数图像如下：

```python
import numpy as np
import matplotlib.pyplot as plt

# 生成一些数据
z = np.linspace(-10, 10, 400)
sigmoid = 1 / (1 + np.exp(-z))

# 绘制 Sigmoid 函数
plt.figure(figsize=(8, 6))
plt.plot(z, sigmoid)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.legend(['Sigmoid'])
plt.show()
```

![](/17.png)
*Fig.17*

## 算法原理

简单来说，逻辑回归做了以下事情：

1.  **线性组合**：将输入特征进行线性加权组合。
2.  **Sigmoid 转换**：将线性组合的结果通过 Sigmoid 函数，映射到 0 到 1 之间的概率值。
3.  **概率预测**：根据概率值，判断样本属于哪个类别。

## 如何使用 scikit-learn 中的 Logistic Regression

### 1. 导入 LogisticRegression 类

```python
from sklearn.linear_model import LogisticRegression
```

### 2. 创建 LogisticRegression 对象

```python
# 创建一个 LogisticRegression 对象
# 可以设置一些参数，比如正则化强度 C，solver 等
model = LogisticRegression(C=1.0, solver='liblinear')
```

常用参数：

*   `C`: 正则化强度的倒数。C 越大，正则化越弱；C 越小，正则化越强。
*   `solver`:  选择优化算法。常用的有 'liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga'。
*   `penalty`:  指定正则化类型。'l1' (L1 正则化), 'l2' (L2 正则化), 'elasticnet', 'none' (无正则化)。
*   `max_iter`: 最大迭代次数。

### 3. 准备数据

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 生成一些示例数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2, random_state=42)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 4. 训练模型

```python
# 使用训练数据拟合模型
model.fit(X_train, y_train)
```

### 5. 预测

```python
# 使用模型进行预测
y_pred = model.predict(X_test)

# 预测概率
y_prob = model.predict_proba(X_test)
```

`predict` 方法给出的是类别预测，而 `predict_proba` 方法给出的是属于每个类别的概率。

### 6. 评估模型

```python
from sklearn.metrics import accuracy_score, classification_report

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 打印分类报告
print(classification_report(y_test, y_pred))

# Accuracy: 0.9666666666666667
#               precision    recall  f1-score   support
# 
#            0       1.00      0.94      0.97        16
#            1       0.93      1.00      0.97        14
# 
#     accuracy                           0.97        30
#    macro avg       0.97      0.97      0.97        30
# weighted avg       0.97      0.97      0.97        30

```

## 完整示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

# 1. 数据准备
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 模型训练
model = LogisticRegression(C=1.0, solver='liblinear')
model.fit(X_train, y_train)

# 3. 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 4. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# 5. 可视化决策边界
# 生成网格数据
h = .02  # 网格步长
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格数据
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)

# 绘制散点图
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu, marker='x', edgecolors='k', label='Testing Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend(loc='best')
plt.show()
```

![](/18.png)
*Fig.18*


## 优点

*   简单易懂，容易实现。
*   计算速度快，适合处理大规模数据。
*   可以直接输出概率值，方便进行决策。
*   可以通过正则化防止过拟合。

## 缺点

*   容易欠拟合，需要进行特征工程。
*   对多重共线性比较敏感。
*   不适合处理非线性可分的数据。

## 使用场景

*   垃圾邮件分类
*   欺诈检测
*   用户点击率预测
*   疾病诊断


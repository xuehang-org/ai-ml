---
title: 常用算法 决策树
---

## 常用算法 决策树

决策树是一种非常流行的机器学习算法，它通过学习一系列的 `if-then-else` 规则来进行决策。你可以把它想象成一个流程图，每个节点代表一个特征的判断，每个分支代表一个判断结果，最终到达叶子节点，叶子节点代表预测结果。

### 1. 决策树的基本概念

*   **节点 (Node):** 决策树中的每个判断点。
*   **根节点 (Root Node):** 树的顶端，第一个进行判断的节点。
*   **叶节点 (Leaf Node):** 树的末端，代表最终的预测结果。
*   **分支 (Branch):** 连接节点的线，代表判断的结果。
*   **深度 (Depth):** 从根节点到叶节点的最长路径的长度。

### 2. 决策树的原理

决策树的核心在于如何选择最佳的特征来进行分割。选择最佳特征的目标是使得分割后的数据更加“纯净”，也就是让属于同一类别的样本尽可能地分到同一个子节点。

常用的衡量“纯净度”的指标有：

*   **信息增益 (Information Gain):** 基于信息熵 (Entropy) 的概念，选择使得信息增益最大的特征进行分割。
*   **基尼系数 (Gini Impurity):**  表示一个节点中样本类别的不确定性，选择使得基尼系数下降最多的特征进行分割。

### 3. 决策树的类型

*   **分类树 (DecisionTreeClassifier):** 用于处理分类问题，预测样本的类别。
*   **回归树 (DecisionTreeRegressor):** 用于处理回归问题，预测样本的数值。

### 4. 如何使用 scikit-learn 中的决策树

scikit-learn 提供了 `DecisionTreeClassifier` 和 `DecisionTreeRegressor` 类来实现决策树算法。

#### 4.1 分类树 (DecisionTreeClassifier)

**示例：使用 `DecisionTreeClassifier` 对鸢尾花数据集进行分类**

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建 DecisionTreeClassifier 对象
dtc = DecisionTreeClassifier(random_state=42) # 可以设置一些参数，比如树的最大深度

# 4. 训练模型
dtc.fit(X_train, y_train)

# 5. 预测
y_pred = dtc.predict(X_test)

# 6. 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Accuracy: 1.0

```

**代码解释：**

1.  **加载数据集：** 使用 `load_iris()` 函数加载鸢尾花数据集。
2.  **划分数据集：** 使用 `train_test_split()` 函数将数据集划分为训练集和测试集。
3.  **创建模型：** 创建 `DecisionTreeClassifier` 对象，`random_state` 参数用于设置随机种子，保证结果的可重复性。
4.  **训练模型：** 使用 `fit()` 方法在训练集上训练模型。
5.  **预测：** 使用 `predict()` 方法在测试集上进行预测。
6.  **评估模型：** 使用 `accuracy_score()` 函数计算预测的准确率。

#### 4.2 回归树 (DecisionTreeRegressor)

**示例：使用 `DecisionTreeRegressor` 预测房价**

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. 创建模拟数据集
X = np.linspace(0, 10, 100).reshape(-1, 1) # 特征
y = np.sin(X) + np.random.normal(0, 0.1, 100).reshape(-1, 1) # 目标值

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建 DecisionTreeRegressor 对象
dtr = DecisionTreeRegressor(random_state=42)

# 4. 训练模型
dtr.fit(X_train, y_train)

# 5. 预测
y_pred = dtr.predict(X_test)

# 6. 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Mean Squared Error: 0.02520696059160534
```

**代码解释：**

1.  **创建模拟数据集：**  这里我们创建了一个简单的正弦函数数据集，并添加了一些噪声。
2.  **划分数据集：**  使用 `train_test_split()` 函数将数据集划分为训练集和测试集。
3.  **创建模型：**  创建 `DecisionTreeRegressor` 对象。
4.  **训练模型：**  使用 `fit()` 方法在训练集上训练模型。
5.  **预测：**  使用 `predict()` 方法在测试集上进行预测。
6.  **评估模型：**  使用 `mean_squared_error()` 函数计算均方误差。

#### 4.3 可视化决策树

我们可以使用 `sklearn.tree.plot_tree` 函数来可视化决策树。

```python
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

# 1. 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 2. 创建 DecisionTreeClassifier 对象
dtc = tree.DecisionTreeClassifier(max_depth=2, random_state=42) # 限制树的最大深度，避免过度拟合

# 3. 训练模型
dtc.fit(X, y)

# 4. 可视化决策树
plt.figure(figsize=(12, 8))  # 设置图像大小
tree.plot_tree(dtc,
               feature_names=iris.feature_names,
               class_names=[str(c) for c in iris.target_names], # 转换为字符串
               filled=True,
               fontsize=10)  # 调整字体大小
plt.title("Decision Tree Visualization") # 添加标题
plt.show()
```

![](/21.png)
*Fig.21*

**代码解释：**

1.  **加载数据集：** 使用 `load_iris()` 函数加载鸢尾花数据集。
2.  **创建模型：** 创建 `DecisionTreeClassifier` 对象，`max_depth` 参数用于限制树的最大深度，防止过拟合。
3.  **训练模型：** 使用 `fit()` 方法在数据集上训练模型。
4.  **可视化决策树：**  使用 `tree.plot_tree()` 函数将决策树可视化。
    *   `feature_names` 参数指定特征的名称。
    *   `class_names` 参数指定类别的名称。
    *   `filled=True`  表示用颜色填充节点，颜色代表不同的类别。

    如果您的环境中无法显示中文，请确保您已安装合适的字体，并在代码中设置字体。  或者将`class_names`转换为英文，避免显示问题。

### 5. 决策树的优缺点

**优点：**

*   **易于理解和解释：** 决策树的结构清晰，可以很容易地理解其决策过程。
*   **可以处理各种类型的数据：**  包括数值型和类别型数据。
*   **不需要进行特征缩放：**  因为决策树是基于特征的判断来进行分割的。
*   **可以处理缺失值：**  可以通过一些策略来处理缺失值。

**缺点：**

*   **容易过拟合：**  如果树的深度过大，容易在训练集上表现良好，但在测试集上表现较差。
*   **对数据的微小变化敏感：** 数据的微小变化可能导致树的结构发生很大的变化。
*   **容易产生有偏的树：**  如果某些特征在数据集中占主导地位，容易导致树偏向这些特征。

### 6. 决策树的参数调优

为了防止决策树过拟合，我们需要对一些参数进行调优。常用的参数有：

*   `max_depth`:  树的最大深度。
*   `min_samples_split`:  一个节点至少需要多少个样本才能继续分割。
*   `min_samples_leaf`:  一个叶节点至少需要多少个样本。
*   `max_features`:  选择最佳分割特征时，考虑的最大特征数量。

可以使用 `GridSearchCV` 或 `RandomizedSearchCV` 等方法来进行参数调优。

### 7. 总结

决策树是一种简单而强大的机器学习算法，它易于理解和解释，可以处理各种类型的数据。但是，决策树容易过拟合，需要进行参数调优。掌握决策树的原理和使用方法，可以帮助你更好地解决实际问题。

希望这个教程能够帮助你理解和使用 scikit-learn 中的决策树算法！
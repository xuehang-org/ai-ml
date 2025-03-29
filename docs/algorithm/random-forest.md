---
title: 常用算法 随机森林
---

# 常用算法 随机森林

随机森林是一种强大的集成学习算法，在机器学习领域被广泛应用。它属于监督学习算法，即可用于分类，也可用于回归。 随机森林通过构建多个决策树，并以投票或平均的方式进行预测，从而提高整体的准确性和稳定性。

## 核心思想

随机森林的核心思想是“集成学习”。 简单来说，就是“三个臭皮匠，顶个诸葛亮”。 通过组合多个弱学习器（决策树），构建一个强学习器。

其主要思想可以概括为以下几点：

1.  **随机性**：
    *   **数据随机选择**： 从原始训练集中随机选择一部分样本，用于训练每个决策树。 这种方法称为**自助采样（bootstrap sampling）**。
    *   **特征随机选择**： 在每个节点分裂时，不是考虑所有特征，而是随机选择一部分特征，从中选择最优的特征进行分裂。

2.  **集成**：
    *   **多个决策树**： 构建大量的决策树。
    *   **投票/平均**： 分类问题采用投票的方式，回归问题采用平均的方式，综合所有决策树的预测结果。

## 算法原理

1.  **构建多棵决策树**：
    *   **自助采样**： 从原始训练集中有放回地随机抽取 *N* 个样本，作为每棵决策树的训练集。  这样可以保证每棵树的训练数据都有所不同。
    *   **特征选择**： 在每个节点，随机选择 *m* 个特征（*m* < 总特征数），然后从中选择最优的特征进行分裂。  这可以降低树与树之间的相关性，提高整体的泛化能力。
    *   **完全生长**： 每棵树都尽可能地生长，不对其进行剪枝（或者进行较少的剪枝）。

2.  **预测**：
    *   **分类**： 对于分类问题，每棵树都给出自己的预测类别，最终的预测结果是所有树投票最多的类别。
    *   **回归**： 对于回归问题，每棵树都给出自己的预测值，最终的预测结果是所有树预测值的平均值。

## 优点

*   **高准确率**： 通过集成多个决策树，随机森林通常具有很高的准确率。
*   **抗过拟合能力强**： 随机选择数据和特征，降低了模型的方差，不容易过拟合。
*   **可处理高维数据**： 不需要进行特征选择，可以处理大量的特征。
*   **可评估特征重要性**： 可以评估每个特征在预测中的重要性，帮助理解数据。
*   **易于并行化**： 每棵树的训练是独立的，可以并行进行，提高训练效率。

## 缺点

*   **模型复杂度高**： 相比于单个决策树，随机森林的模型复杂度更高，需要更多的计算资源。
*   **可解释性较差**： 相比于单个决策树，随机森林的可解释性较差，难以理解每棵树的具体决策过程。
*   **对于小样本数据，可能表现不佳**： 随机森林需要大量的样本进行训练，对于小样本数据，可能表现不佳。

##  `scikit-learn` 中的随机森林

在 `scikit-learn` 中， 随机森林的实现主要包括两个类：

*   `RandomForestClassifier`： 用于分类问题。
*   `RandomForestRegressor`： 用于回归问题。

### `RandomForestClassifier`

#### 示例： 使用 `RandomForestClassifier` 进行分类

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# 1. 创建模拟数据
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # n_estimators: 决策树的数量

# 4. 训练模型
rf_classifier.fit(X_train, y_train)

# 5. 预测
y_pred = rf_classifier.predict(X_test)

# 6. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 7. 特征重要性
feature_importances = rf_classifier.feature_importances_
print("Feature Importances:", feature_importances)
# Accuracy: 0.8566666666666667
# Feature Importances: [0.01404285 0.10565003 0.01664851 0.01319296 0.01499015 0.33758016
#  0.01953689 0.01805995 0.01606389 0.01464691 0.01579867 0.02492017
#  0.02262103 0.01864569 0.07209489 0.01899313 0.01736364 0.01218901
#  0.21095922 0.01600223]
```

**代码解释**

1.  **创建模拟数据**：  使用 `make_classification` 创建一个包含 1000 个样本，20 个特征的分类数据集。
2.  **分割数据集**： 将数据集分割为训练集和测试集，测试集占 30%。
3.  **创建随机森林分类器**： 创建一个 `RandomForestClassifier` 对象，设置 `n_estimators=100`，表示使用 100 棵决策树。  `random_state` 用于控制随机种子，保证实验的可重复性。
4.  **训练模型**： 使用训练集训练随机森林模型。
5.  **预测**： 使用训练好的模型对测试集进行预测。
6.  **评估**： 使用 `accuracy_score` 计算预测的准确率。
7.  **特征重要性**：  `feature_importances_` 属性可以获取每个特征的重要性得分。

#### 常用参数

*   `n_estimators`： 森林中决策树的数量。  值越大，通常性能越好，但计算成本也会增加。  默认值是 100。
*   `criterion`：  用于衡量分裂质量的函数。  可以是 "gini" (基尼系数) 或 "entropy" (信息增益)。  默认值是 "gini"。
*   `max_depth`：  树的最大深度。  可以控制模型的复杂度，防止过拟合。  如果为 `None`，则树会生长到所有叶子都是纯的，或者所有叶子包含的样本数都小于 `min_samples_split`。
*   `min_samples_split`： 分裂一个节点所需的最小样本数。  可以控制模型的复杂度，防止过拟合。  默认值是 2。
*   `min_samples_leaf`：  一个叶节点所需的最小样本数。  可以控制模型的复杂度，防止过拟合。 默认值是 1。
*   `max_features`：  在寻找最佳分裂时要考虑的特征数量。  可以是 "auto" (等于 `sqrt(n_features)`)、"sqrt"、"log2"、`None` (等于 `n_features`)，或者一个整数或浮点数。
*   `bootstrap`：  是否使用自助采样。  默认值为 `True`。
*   `random_state`： 随机种子，用于控制随机性，保证实验的可重复性。

### `RandomForestRegressor`

#### 示例： 使用 `RandomForestRegressor` 进行回归

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. 创建模拟数据
X, y = make_regression(n_samples=100, n_features=1, noise=5, random_state=42)

# 2. 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建随机森林回归器
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. 训练模型
rf_regressor.fit(X_train, y_train)

# 5. 预测
y_pred = rf_regressor.predict(X_test)

# 6. 评估
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 7. 可视化
plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, y_pred, label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Random Forest Regression")
plt.legend()
plt.show()
```

![](/22.png)
*Fig.22*

**代码解释**

1.  **创建模拟数据**： 使用 `make_regression` 创建一个包含 100 个样本，1 个特征的回归数据集。
2.  **分割数据集**： 将数据集分割为训练集和测试集，测试集占 30%。
3.  **创建随机森林回归器**： 创建一个 `RandomForestRegressor` 对象，设置 `n_estimators=100`，表示使用 100 棵决策树。
4.  **训练模型**： 使用训练集训练随机森林模型。
5.  **预测**： 使用训练好的模型对测试集进行预测。
6.  **评估**： 使用 `mean_squared_error` 计算预测的均方误差。
7.  **可视化**： 将实际值和预测值绘制在同一张图上，进行比较。

#### 常用参数

`RandomForestRegressor` 的常用参数与 `RandomForestClassifier` 基本相同，除了 `criterion` 参数。

*   `criterion`： 用于衡量分裂质量的函数。  可以是 "mse" (均方误差)，"mae" (平均绝对误差) 或 "poisson"。  默认值是 "mse"。

## 特征重要性

随机森林可以评估每个特征在预测中的重要性。  其基本思想是：如果一个特征在很多树中都被用于分裂节点，那么这个特征就比较重要。

可以通过 `feature_importances_` 属性获取特征重要性得分。

```python
# 在训练模型后
feature_importances = rf_classifier.feature_importances_ # 或 rf_regressor.feature_importances_
print("Feature Importances:", feature_importances)
```

特征重要性得分越高，表示该特征在预测中越重要。  可以利用特征重要性进行特征选择，选择重要的特征进行建模，提高模型的效率和泛化能力。

## 总结

随机森林是一种强大的集成学习算法，具有高准确率、抗过拟合能力强等优点。  在 `scikit-learn` 中，可以使用 `RandomForestClassifier` 和 `RandomForestRegressor` 分别进行分类和回归。  通过调整参数，可以优化模型的性能。 此外，随机森林还可以评估特征的重要性，帮助理解数据和进行特征选择。


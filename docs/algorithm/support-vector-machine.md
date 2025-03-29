---
title: 常用算法 支持向量机
---

# 常用算法 支持向量机

支持向量机 (SVM) 是一种强大的**分类**和**回归**算法。它试图找到一个最佳的**超平面**，能够最大程度地区分不同类别的数据。你可以把它想象成在两类数据之间画一条线（或者一个面、一个超平面），让这条线离两边的数据都尽可能远。

## 核心概念

*   **超平面 (Hyperplane)**： 在 SVM 中，超平面是用来分隔不同类别数据的决策边界。 在二维空间中，超平面就是一条直线；在三维空间中，超平面就是一个平面；在更高维空间中，它就是一个“超平面”。
*   **支持向量 (Support Vectors)**： 它们是距离超平面最近的那些数据点。 SVM 的训练目标就是找到能够最大化这些支持向量到超平面的距离的超平面。
*   **间隔 (Margin)**： 间隔是指超平面与最近的支持向量之间的距离。 SVM 的目标是最大化这个间隔，因为更大的间隔通常意味着更好的泛化能力。

## SVM 的类型

Scikit-learn 提供了多种 SVM 的实现，主要包括：

*   **`SVC` (Support Vector Classification)**： 用于分类问题。
*   **`NuSVC`**： 类似于 SVC，但使用不同的参数控制支持向量的数量。
*   **`LinearSVC`**： 用于大规模线性分类问题，速度更快。
*   **`SVR` (Support Vector Regression)**： 用于回归问题。
*   **`NuSVR`**： 类似于 SVR，但使用不同的参数控制支持向量的数量。
*   **`OneClassSVM`**： 用于异常检测。

## 线性 SVM (Linear SVM)

当数据是线性可分的时候，我们可以使用线性 SVM。

### 示例：线性分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 创建模拟数据
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# 创建 SVM 分类器
clf = svm.SVC(kernel='linear', C=1)  # C 是正则化参数
clf.fit(X, y)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 画决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画出决策边界和间隔
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# 突出显示支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('Linear SVM')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

![](/19.png)
*Fig.19*


**代码解释：**

1.  `make_blobs` 创建了两类数据，每类 50 个样本。
2.  `svm.SVC(kernel='linear', C=1)` 创建了一个线性 SVM 分类器。 `kernel='linear'` 指定使用线性核函数。 `C` 是正则化参数，控制对错误分类的惩罚程度。
3.  `clf.fit(X, y)` 训练模型。
4.  后面的代码用于可视化决策边界、间隔和支持向量。

## 非线性 SVM (Non-linear SVM)

当数据不是线性可分的时候，我们需要使用非线性 SVM。 这时，SVM 会使用**核函数**将数据映射到更高维的空间，然后在高维空间中找到一个超平面来分隔数据。

### 常用核函数

*   **`rbf` (径向基函数)**： 默认的核函数，适用于大多数情况。
*   **`poly` (多项式核函数)**： 适用于数据分布较为规则的情况。
*   **`sigmoid` (sigmoid 核函数)**： 类似于神经网络中的 sigmoid 函数。

### 示例：非线性分类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons

# 创建非线性可分的数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=0)

# 创建 SVM 分类器 (RBF 核)
clf = svm.SVC(kernel='rbf', gamma=0.5, C=1) # gamma 影响 RBF 核的形状
clf.fit(X, y)

# 可视化
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# 画决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 画出决策边界和间隔
ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

# 突出显示支持向量
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.title('Non-linear SVM (RBF Kernel)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

![](/20.png)
*Fig.20*

**代码解释：**

1.  `make_moons` 创建了一个非线性可分的数据集，形状像两个交错的月亮。
2.  `svm.SVC(kernel='rbf', gamma=0.5, C=1)` 创建了一个使用 RBF 核的 SVM 分类器。 `gamma` 参数影响 RBF 核的形状，`C` 是正则化参数。
3.  后面的代码用于可视化决策边界和支持向量。

## 参数调优

SVM 的性能高度依赖于参数的选择。 一些重要的参数包括：

*   **`C` (正则化参数)**： 控制对错误分类的惩罚程度。 `C` 越大，模型越倾向于正确分类所有训练样本，但也可能导致过拟合。 `C` 越小，模型越容忍错误分类，可能导致欠拟合。
*   **`kernel` (核函数)**： 选择合适的核函数对性能至关重要。
*   **`gamma` (核系数)**： 仅对 `rbf`、`poly` 和 `sigmoid` 核有效。 它控制核函数的形状。 `gamma` 越大，模型越复杂，容易过拟合。

可以使用网格搜索 (`GridSearchCV`) 或随机搜索 (`RandomizedSearchCV`) 来找到最佳参数。

### 示例：网格搜索

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 定义参数网格
param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 'scale'],
              'kernel': ['rbf']}

# 创建 GridSearchCV 对象
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)

# 训练模型
grid.fit(X_train, y_train)

# 打印最佳参数
print("Best parameters:", grid.best_params_)

# 评估模型
print("Accuracy:", grid.score(X_test, y_test))

# Fitting 5 folds for each of 16 candidates, totalling 80 fits
# [CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=0.1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=0.1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=0.1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=0.1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ........................C=1, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=1, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ...........................C=1, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .......................C=1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .......................C=1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .......................C=1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .......................C=1, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=10, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ........................C=10, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ..........................C=10, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ......................C=10, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END ......................C=100, gamma=0.01, kernel=rbf; total time=   0.0s
# [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .......................C=100, gamma=0.1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .........................C=100, gamma=1, kernel=rbf; total time=   0.0s
# [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
# [CV] END .....................C=100, gamma=scale, kernel=rbf; total time=   0.0s
# Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
# Accuracy: 0.9777777777777777
```

**代码解释：**

1.  `load_iris` 加载鸢尾花数据集。
2.  `train_test_split` 将数据集分割为训练集和测试集。
3.  `param_grid` 定义了要搜索的参数网格。
4.  `GridSearchCV` 创建了一个网格搜索对象。 `refit=True` 表示在找到最佳参数后，使用整个训练集重新训练模型。 `verbose=2` 表示打印详细的训练信息。
5.  `grid.fit(X_train, y_train)` 训练模型，网格搜索会自动尝试所有参数组合。
6.  `grid.best_params_` 打印最佳参数。
7.  `grid.score(X_test, y_test)` 评估模型在测试集上的性能。

## 优点和缺点

**优点：**

*   在高维空间中有效。
*   当特征数量大于样本数量时仍然有效。
*   在决策函数中使用支持向量，因此具有内存效率。
*   通用性强： 可以选择不同的核函数来适应不同的数据分布。

**缺点：**

*   如果特征数量远大于样本数量，性能可能较差。
*   对参数选择和核函数的选择敏感。
*   对于大规模数据集，训练时间可能较长。
*   不易解释： SVM 是一个“黑盒”模型，不容易理解其内部工作原理。

## 总结

SVM 是一种强大的机器学习算法，适用于分类和回归问题。 通过选择合适的核函数和参数，可以获得很好的性能。 但是，也需要注意 SVM 的缺点，并根据具体情况选择合适的算法。

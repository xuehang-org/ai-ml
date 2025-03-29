---
title: 基本概念 无监督学习
---


# 基本概念 无监督学习

无监督学习是机器学习的一个重要分支，它与监督学习最大的区别在于，**训练数据没有标签**。这意味着算法需要自己去发现数据中的模式、结构和关系。

## 核心概念

*   **无标签数据**: 训练数据集中只有特征，没有目标变量。
*   **模式发现**: 算法的目标是从数据中发现隐藏的结构，例如聚类、降维等。
*   **探索性分析**: 无监督学习常用于数据探索，帮助我们理解数据的内在性质。

## 主要类型

### 1. 聚类 (Clustering)

**定义**: 将相似的数据点归为一类，形成不同的簇 (cluster)。

**目标**: 使得簇内数据点尽可能相似，簇间数据点尽可能不同。

**常用算法**:

*   K-均值聚类 (K-Means)
*   层次聚类 (Hierarchical Clustering)
*   DBSCAN

**示例：K-均值聚类**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 创建模拟数据
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 使用K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=200, color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```
![](/2.png)
*Fig.2*

**代码解释**:

1.  `make_blobs` 创建了4个簇的模拟数据。
2.  `KMeans(n_clusters=4)` 创建一个K-Means模型，指定簇的数量为4。
3.  `fit_predict(X)` 对数据进行聚类，并返回每个数据点的簇标签。
4.  `plt.scatter` 用于绘制散点图，`c=y_pred` 指定颜色，`cmap='viridis'` 指定颜色映射。
5.  `kmeans.cluster_centers_` 包含每个簇的中心点坐标，用红色星号标记。

### 2. 降维 (Dimensionality Reduction)

**定义**: 在保留数据关键信息的前提下，减少数据的维度。

**目标**: 简化数据表示，提高计算效率，去除噪声，便于可视化。

**常用算法**:

*   主成分分析 (PCA)
*   t-分布邻域嵌入 (t-SNE)
*   线性判别分析 (LDA)  (虽然LDA常用于监督学习，但在某些场景下也可用于无监督降维)

**示例：PCA降维**

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 可视化结果
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.title('PCA Dimensionality Reduction')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target')
plt.show()
```
![](/3.png)
*Fig.3*

**代码解释**:

1.  `load_iris` 加载经典的鸢尾花数据集。
2.  `PCA(n_components=2)` 创建一个PCA模型，指定降维到2维。
3.  `fit_transform(X)` 对数据进行降维，并返回降维后的数据。
4.  可视化降维后的数据，颜色表示不同的鸢尾花类别。

### 3. 异常检测 (Anomaly Detection)

**定义**: 识别数据集中与其他数据点显著不同的异常值 (outliers)。

**目标**: 发现欺诈行为、设备故障、网络攻击等异常事件。

**常用算法**:

*   孤立森林 (Isolation Forest)
*   局部离群因子 (Local Outlier Factor, LOF)
*   One-Class SVM

**示例：孤立森林异常检测**

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest

# 创建模拟数据，包含一些异常值
rng = np.random.RandomState(42)
X = 0.3 * rng.randn(100, 2)
X = np.r_[X + 2, X - 2]  # 创建两个簇
X = np.r_[X, rng.uniform(low=-4, high=4, size=(20, 2))] # 添加异常值

# 使用孤立森林进行异常检测
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred = iso_forest.fit_predict(X)

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Anomaly Score')
plt.show()
```
![](/4.png)
*Fig.4*
**代码解释**:

1.  创建包含两个簇和一些随机异常值的模拟数据。
2.  `IsolationForest(contamination=0.1)` 创建一个孤立森林模型，`contamination` 参数指定异常值的比例。
3.  `fit_predict(X)` 对数据进行异常检测，返回每个数据点的异常值标签 (1表示正常，-1表示异常)。
4.  可视化结果，不同的颜色表示正常点和异常点。

## 应用场景

*   **市场细分**:  将客户划分为不同的群体，以便进行精准营销。
*   **图像分割**:  将图像分割成不同的区域，用于目标识别和场景理解。
*   **推荐系统**:  根据用户行为，发现相似的用户或物品，进行个性化推荐。
*   **网络安全**:  检测网络流量中的异常行为，防止入侵和攻击。
*   **金融风控**:  识别信用卡欺诈、洗钱等非法活动。

## 总结

无监督学习是一种强大的工具，可以帮助我们从无标签数据中发现有用的信息。 掌握无监督学习的基本概念和常用算法，可以为解决实际问题提供新的思路和方法。

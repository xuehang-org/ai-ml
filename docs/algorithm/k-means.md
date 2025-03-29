---
title: 常用算法 K-Means
---


# 常用算法 K-Means

K-Means 是一种非常流行的无监督学习算法，用于将数据集分成不同的簇（cluster）。它的目标是找到数据中隐藏的类别，无需提前标记数据。

## 算法原理

K-Means 算法的核心思想是：

1.  **选择中心点**：首先，随机选择 K 个点作为初始的簇中心点（K 是你预先设定的簇的数量）。
2.  **分配数据点**：对于数据集中的每个点，计算它与每个簇中心点的距离，并将该点分配到距离最近的簇。
3.  **更新中心点**：重新计算每个簇的中心点，通常是计算簇中所有点的平均值。
4.  **迭代**：重复步骤 2 和 3，直到簇的分配不再发生变化，或者达到预设的最大迭代次数。

## 如何使用 Scikit-learn 中的 K-Means

Scikit-learn 提供了 `KMeans` 类来实现 K-Means 算法。下面是如何使用它的基本步骤：

1.  **导入 `KMeans` 类**：

    ```python
    from sklearn.cluster import KMeans
    ```

2.  **创建 `KMeans` 对象**：

    ```python
    kmeans = KMeans(n_clusters=3, random_state=0, n_init = 'auto') # 指定簇的数量 K
    ```

    *   `n_clusters`:  簇的数量，也就是你想把数据分成几类。
    *   `random_state`:  一个随机种子，用于初始化簇中心点。设置它可以保证每次运行结果一致。
    *    `n_init`: ‘auto’ or int, default=‘auto’，设置运行kmeans算法的次数，选择其中最好的一次，默认auto会自己选择一个比较合适的值。

3.  **拟合数据**：

    ```python
    kmeans.fit(data)  # data 是你的数据
    ```

4.  **获取聚类结果**：

    ```python
    labels = kmeans.labels_  # 每个数据点所属的簇的标签
    centers = kmeans.cluster_centers_  # 簇中心点
    ```

## 示例

### 示例 1：简单数据聚类

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 创建模拟数据
data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 创建 KMeans 对象
kmeans = KMeans(n_clusters=4, random_state=0, n_init = 'auto')

# 拟合数据
kmeans.fit(data)

# 获取聚类标签和中心点
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', label='Data Points')
plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='red', s=200, label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

![](/26.png)
*Fig.26*

在这个例子中，我们首先使用 `make_blobs` 创建了一些模拟数据，然后使用 K-Means 将数据分成 4 个簇，并可视化了聚类结果和簇中心点。

### 示例 2：使用 K-Means 进行图像压缩

K-Means 还可以用于图像压缩。其基本思想是将图像中的颜色视为数据点，然后使用 K-Means 将颜色分成 K 个簇。每个像素的颜色值用其所属簇的中心颜色值代替，从而减少图像中颜色的数量，达到压缩的目的。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io

# 加载图像
image = io.imread('data/images/flower.jpg')  # 替换成你的图像路径
image = image / 255  # 归一化像素值到 0-1 之间
original_shape = image.shape

# 将图像转换为二维数组，每个像素点包含 R, G, B 三个颜色通道
data = image.reshape((-1, 3))

# 使用 K-Means 聚类颜色
kmeans = KMeans(n_clusters=16, random_state=0, n_init = 'auto')  # 将颜色聚类成 16 个簇
kmeans.fit(data)

# 获取聚类标签和中心颜色
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 将每个像素点的颜色替换为所属簇的中心颜色
compressed_data = centers[labels]

# 将压缩后的数据转换回原始图像的形状
compressed_image = compressed_data.reshape(original_shape)

# 显示原始图像和压缩后的图像
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[1].imshow(compressed_image)
ax[1].set_title('Compressed Image (16 colors)')
plt.show()
```

![](/27.png)
*Fig.27*

这个例子展示了如何使用 K-Means 减少图像中的颜色数量，从而实现图像压缩。

## 注意事项

*   **选择 K 值**：K-Means 算法需要预先指定簇的数量 K。选择合适的 K 值是一个挑战。常用的方法包括肘部法则（Elbow Method）和轮廓系数（Silhouette Score）。
*   **数据标准化**：K-Means 算法对数据的尺度很敏感。如果数据的不同特征尺度差异很大，建议先进行标准化处理（例如使用 `StandardScaler`）。
*   **初始中心点**：K-Means 算法的初始中心点是随机选择的，不同的初始中心点可能导致不同的聚类结果。可以多次运行 K-Means 算法，选择效果最好的一次。可以通过设置`n_init`参数来设置运行的次数。
*   **局部最优解**：K-Means 算法可能收敛到局部最优解，而不是全局最优解。

希望这个文档能够帮助你理解和使用 K-Means 算法。

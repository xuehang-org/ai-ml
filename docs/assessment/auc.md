---
title: 模型评估 AUC
---

# 模型评估 AUC

## 什么是AUC？

AUC，全称 Area Under the Curve，中文译为“曲线下面积”。它是一种用于评估二分类模型性能的重要指标，特别是当你的数据集类别不平衡时。AUC 衡量的是分类器区分正类和负类的能力。简单来说，AUC 值越高，模型区分正负样本的能力就越强。

想象一下，你有一个模型可以预测用户是否会点击广告。如果模型能够很好地区分会点击和不会点击的用户，那么它的 AUC 值就会很高。

## 为什么要用AUC？

*   **类别不平衡时的可靠指标**：在正负样本比例悬殊的情况下，准确率可能会产生误导。AUC 对此不敏感。
*   **概率输出的有效评估**：AUC 基于模型输出的概率值进行评估，而不是仅仅依赖于硬性的类别判定。
*   **模型排序能力的衡量**：AUC 能够告诉你模型将正样本排在负样本之前的能力有多强。

## AUC的原理

AUC 的计算基于 ROC (Receiver Operating Characteristic) 曲线。ROC 曲线以假正率（False Positive Rate, FPR）为横轴，真正率（True Positive Rate, TPR）为纵轴。

*   **真正率 (TPR)**：也称为灵敏度（Sensitivity）或召回率（Recall），表示所有**实际为正**的样本中，被正确预测为正的比例。
    *   `TPR = TP / (TP + FN)`
*   **假正率 (FPR)**：表示所有**实际为负**的样本中，被错误预测为正的比例。
    *   `FPR = FP / (FP + TN)`

其中：

*   TP (True Positive)：真正例，实际为正，预测为正。
*   FN (False Negative)：假反例，实际为正，预测为负。
*   FP (False Positive)：假正例，实际为负，预测为正。
*   TN (True Negative)：真反例，实际为负，预测为负。

ROC 曲线上的每个点都对应于一个特定的分类阈值。通过调整阈值，你可以得到不同的 FPR 和 TPR，从而绘制出 ROC 曲线。AUC 就是 ROC 曲线下的面积。

**理想情况下**：

*   一个完美的分类器，能将所有正样本排在负样本之前，AUC = 1。
*   一个随机猜测的分类器，AUC = 0.5 (ROC 曲线是一条对角线)。

## 如何在 scikit-learn 中计算 AUC？

scikit-learn 提供了 `roc_auc_score` 函数来计算 AUC。

```python
from sklearn.metrics import roc_auc_score
import numpy as np

# 示例：假设你有一些真实标签 (y_true) 和模型预测的概率值 (y_scores)
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_scores = np.array([0.1, 0.3, 0.6, 0.8, 0.2, 0.7, 0.15, 0.25, 0.85, 0.35])

auc = roc_auc_score(y_true, y_scores)
print("AUC:", auc)
# AUC: 1.0
```

这段代码首先导入了 `roc_auc_score` 函数。然后，我们创建了两个 NumPy 数组：`y_true` 存储真实的类别标签，`y_scores` 存储模型预测的概率值。  `roc_auc_score` 函数会计算出 AUC 值，并将其打印出来。

## 绘制 ROC 曲线

除了计算 AUC，绘制 ROC 曲线也能更直观地了解模型的性能。  scikit-learn 提供了 `roc_curve` 函数来计算 ROC 曲线上的点，然后你可以使用 matplotlib 绘制曲线。

```python
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# 计算 ROC 曲线上的点
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 绘制随机猜测的基线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

![](/29.png)
*Fig.29*

这段代码首先使用 `roc_curve` 函数计算 FPR、TPR 和阈值。然后，使用 matplotlib 绘制 ROC 曲线，并添加 AUC 值作为图例。  还绘制了一条对角线作为随机猜测的基线。

**代码解释:**

1.  `fpr, tpr, thresholds = roc_curve(y_true, y_scores)`:  `roc_curve` 函数返回计算 ROC 曲线所需的数据：假正率 (fpr)、真正率 (tpr) 和对应的阈值。
2.  `plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc)`: 绘制 ROC 曲线，使用蓝色线条，并在图例中显示 AUC 值。
3.  `plt.plot([0, 1], [0, 1], color='gray', linestyle='--')`: 绘制一条从 (0, 0) 到 (1, 1) 的对角线，代表随机猜测的分类器的 ROC 曲线。
4.  `plt.xlim([0.0, 1.0])` 和 `plt.ylim([0.0, 1.05])`:  设置坐标轴的范围，使 ROC 曲线能够完整显示。
5.  `plt.xlabel('False Positive Rate (FPR)')` 和 `plt.ylabel('True Positive Rate (TPR)')`:  设置坐标轴的标签。
6.  `plt.title('Receiver Operating Characteristic (ROC)')`:  设置图表的标题。
7.  `plt.legend(loc="lower right")`:  显示图例，并将图例放在右下角。

这段代码会生成一个 ROC 曲线图，让你直观地了解模型的性能。

## 示例：使用 Logistic Regression 和 AUC 评估模型

下面是一个完整的示例，展示如何使用 Logistic Regression 模型，并使用 AUC 来评估其性能。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 1. 创建一个模拟的二分类数据集
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 2. 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建一个 Logistic Regression 模型
model = LogisticRegression()

# 4. 在训练集上训练模型
model.fit(X_train, y_train)

# 5. 使用模型预测测试集的概率值
y_scores = model.predict_proba(X_test)[:, 1]  # 获取正类的概率

# 6. 计算 AUC
auc = roc_auc_score(y_test, y_scores)
print("AUC:", auc)

# 7. 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
```

![](/30.png)
*Fig.30*

**代码解释:**

1.  **创建数据集**：`make_classification`  函数创建一个模拟的二分类数据集，包含 1000 个样本，20 个特征。
2.  **划分数据集**：`train_test_split`  函数将数据集划分为训练集和测试集，其中测试集占 30%。
3.  **创建模型**：`LogisticRegression`  创建一个 Logistic Regression 模型。
4.  **训练模型**：`model.fit`  在训练集上训练模型。
5.  **预测概率**：`model.predict_proba(X_test)[:, 1]`  预测测试集中每个样本属于正类的概率。  `predict_proba`  返回一个二维数组，其中第一列是属于负类的概率，第二列是属于正类的概率。  我们只取第二列。
6.  **计算 AUC**：`roc_auc_score`  计算 AUC 值。
7.  **绘制 ROC 曲线**：使用前面介绍的方法绘制 ROC 曲线。

## 总结

AUC 是一种强大的模型评估指标，特别是在处理类别不平衡问题时。  通过理解 AUC 的原理和使用 scikit-learn 中的相关函数，你可以更好地评估和比较不同的二分类模型。


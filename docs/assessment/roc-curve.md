---
title: 模型评估 ROC 曲线
---

# 模型评估 ROC 曲线

在机器学习中，模型评估是至关重要的一步。我们训练好模型后，需要知道模型在实际应用中的表现如何。ROC 曲线是一种常用的模型评估工具，尤其是在二分类问题中。

## 什么是 ROC 曲线？

ROC（Receiver Operating Characteristic）曲线，即受试者工作特征曲线。它以假正率（False Positive Rate, FPR）为横轴，真正率（True Positive Rate, TPR）为纵轴，描述了在不同阈值下，分类器的性能表现。

*   **真正率 (TPR, True Positive Rate, Sensitivity, 灵敏度)**：
    $$
    TPR = \frac{TP}{TP + FN}
    $$
    表示所有**实际为正**的样本中，被正确预测为正的比例。

*   **假正率 (FPR, False Positive Rate)**：
    $$
    FPR = \frac{FP}{FP + TN}
    $$
    表示所有**实际为负**的样本中，被错误预测为正的比例。

    其中：

*   TP（True Positive）：实际为正，预测为正。
*   TN（True Negative）：实际为负，预测为负。
*   FP（False Positive）：实际为负，预测为正。
*   FN（False Negative）：实际为正，预测为负。

## 如何绘制 ROC 曲线？

在 scikit-learn 中，我们可以使用 `roc_curve` 函数来计算 ROC 曲线的 FPR 和 TPR，然后使用 Matplotlib 绘制曲线。

### 示例

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

# 1. 创建一个模拟的二分类数据集
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)

# 2. 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建一个 Logistic Regression 模型
model = LogisticRegression()

# 4. 训练模型
model.fit(X_train, y_train)

# 5. 预测测试集的概率
y_prob = model.predict_proba(X_test)[:, 1]

# 6. 计算 ROC 曲线的 FPR 和 TPR
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# 7. 计算 ROC 曲线下的面积 (AUC)
roc_auc = roc_auc_score(y_test, y_prob)

# 8. 绘制 ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

### 代码解释

1.  **创建数据集**：使用 `make_classification` 创建一个包含 1000 个样本的二分类数据集。
2.  **划分数据集**：将数据集划分为训练集和测试集，测试集占 30%。
3.  **创建模型**：使用 `LogisticRegression` 创建一个逻辑回归模型。
4.  **训练模型**：使用训练集训练模型。
5.  **预测概率**：使用训练好的模型预测测试集中每个样本为正类的概率。`predict_proba` 方法返回的是一个二维数组，第一列是负类的概率，第二列是正类的概率。
6.  **计算 ROC 曲线**：使用 `roc_curve` 函数计算 FPR、TPR 和阈值。
7.  **计算 AUC**：使用 `roc_auc_score` 函数计算 ROC 曲线下的面积。AUC 值越大，模型的性能越好。
8.  **绘制 ROC 曲线**：使用 Matplotlib 绘制 ROC 曲线。

### 运行结果

运行上述代码，会得到一张 ROC 曲线图，如下所示：


![](/28.png)
*Fig.28*

图中的橙色曲线就是 ROC 曲线，蓝色的虚线是随机猜测的基准线。ROC 曲线越靠近左上角，模型的性能越好。AUC 值表示 ROC 曲线下的面积，AUC 值越大，模型的性能越好。

## ROC 曲线有什么用？

*   **评估模型性能**：ROC 曲线可以直观地展示模型在不同阈值下的性能表现。
*   **选择最佳阈值**：根据实际需求，选择合适的阈值，以达到最佳的分类效果。
*   **比较不同模型**：可以通过比较不同模型的 ROC 曲线，选择性能最好的模型。

## 总结

ROC 曲线是一种强大的模型评估工具，可以帮助我们更好地了解模型的性能，并选择合适的模型和阈值。希望通过本文的介绍，你能够掌握 ROC 曲线的基本概念和使用方法，并在实际应用中灵活运用。


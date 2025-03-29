---
title: 模型评估 F1 值
---


# 模型评估 F1 值

在机器学习中，我们训练模型是为了让它能够准确地预测新的数据。但是，模型的效果好不好，不能只看它预测对了多少，还要看它有没有漏掉重要的信息。`F1 值` 就是一个综合考虑了这两个方面的指标，它试图找到 `精确率` 和 `召回率` 之间的平衡。

## 1. 什么是 F1 值？

简单来说，`F1 值` 是 `精确率（Precision）` 和 `召回率（Recall）` 的调和平均数。它给 `精确率` 和 `召回率` 赋予了相同的权重，所以当两者都比较高时，`F1 值` 才会比较高。

*   **精确率（Precision）**：预测为正例的样本中，真正是正例的比例。简单说就是“预测对的正例占所有预测为正例的比例”。
*   **召回率（Recall）**：真正是正例的样本中，被预测为正例的比例。简单说就是“预测对的正例占所有真正正例的比例”。

公式如下：

$$
F1 = 2 * \frac{Precision * Recall}{Precision + Recall}
$$

## 2. 为什么要用 F1 值？

假设我们要做一个识别垃圾邮件的模型：

*   **高精确率，低召回率**：模型很保守，只把非常有把握的邮件识别为垃圾邮件。这样可以避免把重要的邮件误判为垃圾邮件，但是可能会有很多垃圾邮件漏网。
*   **低精确率，高召回率**：模型很激进，只要稍微有点像垃圾邮件的都识别出来。这样可以保证不会漏掉垃圾邮件，但是可能会把很多重要的邮件误判为垃圾邮件。

这两种情况都有问题。`F1 值` 试图找到一个平衡点，让我们在避免误判的同时，也尽可能地找出所有应该找出的正例。

## 3. 如何计算 F1 值？

### 3.1 手动计算

首先，我们需要了解混淆矩阵（Confusion Matrix）：

|           | 预测为正例 | 预测为负例 |
|-----------|-------|-------|
| **实际为正例** | TP    | FN    |
| **实际为负例** | FP    | TN    |

*   **TP（True Positive）**：真正例，实际为正例，预测也为正例。
*   **FN（False Negative）**：假反例，实际为正例，预测为负例（漏判）。
*   **FP（False Positive）**：假正例，实际为负例，预测为正例（误判）。
*   **TN（True Negative）**：真反例，实际为负例，预测也为负例。

有了混淆矩阵，我们就可以计算精确率和召回率，然后计算 F1 值：

$$
Precision = \frac{TP}{TP + FP}
$$

$$
Recall = \frac{TP}{TP + FN}
$$

$$
F1 = 2 * \frac{Precision * Recall}{Precision + Recall}
$$

### 3.2 使用 scikit-learn 计算

scikit-learn 提供了 `f1_score` 函数，可以方便地计算 F1 值。

```python
from sklearn.metrics import f1_score

# 真实标签
y_true = [0, 1, 1, 0, 1, 0]
# 预测标签
y_pred = [0, 1, 0, 0, 1, 1]

f1 = f1_score(y_true, y_pred)
print(f"F1 值: {f1}")
# F1 值: 0.6666666666666666
```

## 4. F1 值的应用场景

F1 值在以下场景中非常有用：

*   **不平衡数据集**：当正例和负例的比例相差很大时，准确率（Accuracy）可能会失效，而 F1 值可以更好地反映模型的效果。
*   **需要同时关注精确率和召回率的场景**：例如，在医疗诊断、反欺诈等领域，我们既要避免误判，也要尽可能地找出所有潜在的病例或欺诈行为。
*   **多分类问题**：scikit-learn 提供了 `f1_score` 函数的多个版本，可以用于计算多分类问题的 F1 值，例如 `f1_score(average='macro')`、`f1_score(average='weighted')` 等。

## 5. 示例：使用 F1 值评估模型

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.datasets import make_classification

# 1. 创建一个模拟数据集
X, y = make_classification(n_samples=1000, n_classes=2, weights=[0.9, 0.1], random_state=42) # 创建一个不平衡数据集

# 2. 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 训练一个逻辑回归模型
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# 4. 在测试集上进行预测
y_pred = model.predict(X_test)

# 5. 计算 F1 值
f1 = f1_score(y_test, y_pred)
print(f"F1 值: {f1}")

# 6. 打印分类报告，包含精确率、召回率、F1 值等信息
print(classification_report(y_test, y_pred))

# F1 值: 0.47058823529411764
#               precision    recall  f1-score   support
# 
#            0       0.94      0.97      0.95       270
#            1       0.57      0.40      0.47        30
# 
#     accuracy                           0.91       300
#    macro avg       0.75      0.68      0.71       300
# weighted avg       0.90      0.91      0.90       300
```

在这个例子中，我们创建了一个不平衡的数据集（正例和负例的比例为 9:1），然后训练了一个逻辑回归模型，并使用 F1 值来评估模型的效果。`classification_report` 函数可以提供更详细的评估报告，包括精确率、召回率、F1 值以及支持度（Support，即每个类别在测试集中出现的次数）。

## 6. 多分类 F1 值

对于多分类问题，`f1_score` 函数提供了几种计算 F1 值的平均方法：

*   `average='macro'`：计算每个类别的 F1 值，然后取平均。
*   `average='weighted'`：计算每个类别的 F1 值，然后根据每个类别的样本数量进行加权平均。
*   `average='micro'`：将所有类别的 TP、FP、FN 加起来，然后计算精确率、召回率和 F1 值。
*   `average=None`：返回每个类别的 F1 值。

示例：

```python
from sklearn.metrics import f1_score

# 真实标签
y_true = [0, 1, 2, 0, 1, 2]
# 预测标签
y_pred = [0, 2, 1, 0, 0, 2]

f1_macro = f1_score(y_true, y_pred, average='macro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_none = f1_score(y_true, y_pred, average=None)

print(f"Macro F1 值: {f1_macro}")
print(f"Weighted F1 值: {f1_weighted}")
print(f"Micro F1 值: {f1_micro}")
print(f"每个类别的 F1 值: {f1_none}")

# Macro F1 值: 0.43333333333333335
# Weighted F1 值: 0.43333333333333335
# Micro F1 值: 0.5
# 每个类别的 F1 值: [0.8 0.  0.5]
```

## 7. 总结

`F1 值` 是一个非常有用的模型评估指标，它综合考虑了精确率和召回率，可以帮助我们更好地了解模型的效果，特别是在不平衡数据集和需要同时关注精确率和召回率的场景中。希望这篇文章能够帮助你更好地理解和使用 F1 值。

---
title: 模型评估 召回率
---


# 模型评估 召回率

在机器学习中，尤其是在分类任务中，召回率是一个重要的评估指标。它衡量的是模型能够正确识别出所有**真正例**的能力，即“找回了多少”。

## 定义

召回率（Recall），也称为**查全率**，表示的是所有**实际为正例**的样本中，被模型正确预测为正例的比例。

## 公式

召回率的计算公式如下：

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

其中：

-   **TP (True Positive)**：真正例，即实际为正例且被模型正确预测为正例的样本数。
-   **FN (False Negative)**：假反例，即实际为正例但被模型错误预测为负例的样本数。

分母 `TP + FN` 其实就是所有实际正例的数量。

## 示例

假设我们有一个疾病检测模型，用于判断病人是否患有某种疾病。我们收集了 100 个人的数据，其中 60 个人实际患病（正例），40 个人未患病（负例）。模型预测结果如下：

-   真正例 (TP)：模型正确预测了 50 个患病的人。
-   假反例 (FN)：模型错误地预测了 10 个患病的人为未患病。
-   假正例 (FP)：模型错误地预测了 5 个人为患病，但他们实际上未患病。
-   真反例 (TN)：模型正确预测了 35 个未患病的人。

那么，该模型的召回率计算如下：

$$
\text{Recall} = \frac{50}{50 + 10} = \frac{50}{60} \approx 0.833
$$

这意味着模型能够找出 83.3% 的患病者。

## Scikit-learn 实现

在 Scikit-learn 中，可以使用 `recall_score` 函数来计算召回率。

```python
from sklearn.metrics import recall_score

# 真实标签
y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
# 模型预测标签
y_pred = [1, 1, 1, 0, 1, 0, 0, 0, 1, 0]

# 计算召回率
recall = recall_score(y_true, y_pred)
print(f"召回率: {recall}")
# 召回率: 0.6666666666666666
```

在这个例子中，我们有 6 个正例。模型正确识别了其中的 4 个，但漏掉了 2 个。因此，召回率为 4/6 ≈ 0.67。

## 不同 `average` 参数的影响

`recall_score` 函数还提供了一个 `average` 参数，用于处理多类别分类问题。常用的 `average` 参数有：

-   `'binary'`：**（默认）** 仅适用于二分类问题，计算正例的召回率。如果用于多分类，会报错。
-   `'micro'`：计算所有样本的总 TP 和总 FN，然后计算召回率。
-   `'macro'`：计算每个类别的召回率，然后取平均值（不考虑每个类别的样本数量）。
-   `'weighted'`：计算每个类别的召回率，然后取加权平均值（考虑每个类别的样本数量）。
-   `None`:  返回每个类别的召回率。

示例：

```python
from sklearn.metrics import recall_score

# 多类别真实标签
y_true = [0, 1, 2, 0, 1, 2]
# 多类别模型预测标签
y_pred = [0, 2, 1, 0, 0, 1]

# 计算不同 average 方式下的召回率
# recall_binary = recall_score(y_true, y_pred, average='binary') # 这行代码会报错，因为是多分类问题
recall_micro = recall_score(y_true, y_pred, average='micro')
recall_macro = recall_score(y_true, y_pred, average='macro')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
recall_none = recall_score(y_true, y_pred, average=None)

# print(f"Binary Recall: {recall_binary}")
print(f"Micro Recall: {recall_micro}")
print(f"Macro Recall: {recall_macro}")
print(f"Weighted Recall: {recall_weighted}")
print(f"None Recall: {recall_none}")

# Micro Recall: 0.3333333333333333
# Macro Recall: 0.3333333333333333
# Weighted Recall: 0.3333333333333333
# None Recall: [1. 0. 0.]
```

**注意：**  当处理多分类问题时，不要使用 `average='binary'`，因为它只适用于二分类。  应该选择 `'micro'`, `'macro'`, `'weighted'` 或 `None`  来计算召回率。  `None` 会返回每个类别的召回率，这在分析模型在不同类别上的表现时非常有用。

## 召回率与精确率的权衡

召回率和精确率 (Precision) 是一对重要的指标，通常需要一起考虑。

-   **精确率** 衡量的是所有被模型预测为正例的样本中，有多少是真正的正例。
-   **召回率** 衡量的是所有真正的正例中，有多少被模型正确预测为正例。

在实际应用中，提高召回率可能会降低精确率，反之亦然。例如，一个“宁可错杀一千，绝不放过一个”的模型，会尽可能多地将样本预测为正例，从而提高召回率，但也会导致较高的假正例率，降低精确率。

## 应用场景

召回率在以下场景中尤为重要：

-   **医疗诊断**：尽可能找出所有患病的人，避免漏诊。
-   **反欺诈**：尽可能检测出所有欺诈行为，减少经济损失。
-   **信息检索**：尽可能返回所有相关的文档，避免遗漏。

## 总结

召回率是评估分类模型的重要指标，它衡量的是模型识别真正例的能力。在实际应用中，需要根据具体场景，权衡召回率和精确率，选择合适的模型和阈值。通过 Scikit-learn 提供的 `recall_score` 函数，可以方便地计算召回率，并根据不同的 `average` 参数进行灵活的调整。  在多分类问题中，请务必选择合适的 `average` 参数，避免出现错误。


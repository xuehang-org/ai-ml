---
title: 模型评估 准确率
---


# 模型评估 准确率

准确率是分类问题中最常用，也最直观的评估指标之一。它衡量的是分类器正确分类的样本比例。

## 公式

准确率的计算公式非常简单：

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
$$

或者可以表示为：

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

其中：

*   `TP` (True Positive): 真正例，模型预测为正例，实际也是正例。
*   `TN` (True Negative): 真反例，模型预测为反例，实际也是反例。
*   `FP` (False Positive): 假正例，模型预测为正例，实际是反例 (也称为 Type I 错误)。
*   `FN` (False Negative): 假反例，模型预测为反例，实际是正例 (也称为 Type II 错误)。

## 例子

假设我们有一个二元分类器，用于判断邮件是否为垃圾邮件。经过分类后，我们得到以下结果：

*   总共有100封邮件
*   其中60封是垃圾邮件，40封不是垃圾邮件
*   模型正确地将50封垃圾邮件分类为垃圾邮件 (TP = 50)
*   模型错误地将10封非垃圾邮件分类为垃圾邮件 (FP = 10)
*   模型正确地将30封非垃圾邮件分类为非垃圾邮件 (TN = 30)
*   模型错误地将10封垃圾邮件分类为非垃圾邮件 (FN = 10)

那么，这个分类器的准确率就是：

$$
\text{Accuracy} = \frac{50 + 30}{50 + 30 + 10 + 10} = \frac{80}{100} = 0.8
$$

也就是说，这个分类器有80%的准确率。

## 如何使用 scikit-learn 计算准确率

Scikit-learn 提供了 `accuracy_score` 函数来计算准确率。

```python
from sklearn.metrics import accuracy_score

# 真实标签
y_true = [0, 1, 1, 0, 1, 0]
# 预测标签
y_pred = [0, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")  # 输出：Accuracy: 0.6666666666666666
```

## 示例：使用准确率评估 Logistic Regression 模型

我们使用经典的手写数字识别数据集 (MNIST) 来演示如何使用准确率评估 Logistic Regression 模型。

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载数据集
digits = load_digits()
X, y = digits.data, digits.target

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 Logistic Regression 模型
model = LogisticRegression(max_iter=5000)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
# Accuracy: 0.9685185185185186
```

## 准确率的局限性

准确率虽然简单易懂，但在某些情况下，它可能无法提供全面的评估信息。尤其是在**不平衡数据集**上。

例如，假设我们有一个疾病检测模型，其中：

*   1000个样本中，只有10个患病 (正例)
*   模型将所有样本都预测为健康 (反例)

那么，模型的准确率是 990/1000 = 99%。 尽管准确率很高，但这显然不是一个好的模型，因为它完全忽略了患病的人。

在这种情况下，其他评估指标 (如精确率、召回率、F1 分数) 能提供更有价值的信息。

## 总结

*   准确率是分类问题中最基本的评估指标，衡量分类器正确分类的样本比例。
*   可以使用 `sklearn.metrics.accuracy_score` 计算准确率。
*   准确率简单易懂，但可能在不平衡数据集上产生误导。
*   在评估模型时，应该结合其他评估指标一起考虑。


# Wordle预测模型两阶段实验总结

## 实验概述

本实验按照以下两步流程完成：

1. 基于games embedding数据训练模型
2. 用games embedding数据测试模型

## 数据集信息

* **训练集**: games\_word\_embedding.npz + games\_char\_embedding.npz
* **测试集**: games\_word\_embedding.npz + games\_char\_embedding.npz

## 实验结果

### 阶段3：基于games embedding数据训练

* **训练模型**: stage3\_reinforced\_model.npz

### 阶段4：测试模型

* **测试结果**: MSE = 1.7058, MAE = 0.9523

### 实验总结

* 模型在游戏数据上的测试MAE为 0.9523
* 模型能够有效从embedding数据中学习，具有良好的性能

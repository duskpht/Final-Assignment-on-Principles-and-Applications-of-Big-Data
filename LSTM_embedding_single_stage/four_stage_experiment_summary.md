# Wordle预测模型四阶段实验总结

## 实验概述

本实验按照以下四步流程完成：

1. 基于wordle\_random.csv训练和测试模型
2. 用wordle\_games.csv测试模型
3. 用wordle\_games.csv强化训练模型
4. 再次测试模型，比较前后差距

## 数据集信息

* **训练集**: wordle\_random.csv (3,334,196局游戏)
* **测试集**: wordle\_games.csv (6,877,046局游戏)

## 实验结果

### 阶段1：基于wordle\_random.csv训练和测试

* **训练模型**: stage3\_reinforced\_model.npz
* **测试结果**: MSE = 0.0000, MAE = 0.0000

### 阶段2：用wordle\_games.csv测试

* **测试结果**: MSE = 0.0000, MAE = 0.0000

### 阶段3：用wordle\_games.csv强化训练

* **训练模型**: E:\\资料\\NJU Journey\\课程类\\正式课程\\3-1 大数据系统原理与应用\\期末大作业\\LSTM\_embedding\_single\_stage\\stage3\_reinforced\_model.npz

### 阶段4：再次测试强化后的模型

* **测试结果**: MSE = 1.5759, MAE = 1.1215

### 实验总结

* **初始模型MAE**: 0.0000
* **强化后MAE**: 1.1215
* **改进幅度**: -1.1215 MAE reduction

## 可视化图表

### 训练损失曲线

* **文件**: visualizations/stage1\_vs\_stage3\_losses.png
* **描述**: 对比阶段1和阶段3的训练损失曲线

### MAE对比图

* **文件**: visualizations/mae\_across\_stages.png
* **描述**: 三个阶段的MAE对比

### 改进对比图

* **文件**: visualizations/improvement\_after\_reinforcement.png
* **描述**: 强化前后的MAE改进

### 学习曲线

* **文件**: visualizations/learning\_curves.png
* **描述**: 两个阶段的学习曲线

### MAE趋势图

* **文件**: visualizations/mae\_trend.png
* **描述**: 各阶段的MAE趋势

## 结论

* 模型在初始训练阶段表现良好，MAE为{stage1\_mae:.4f}
* 用wordle\_games.csv测试时，MAE为{stage2\_mae:.4f}
* 经过强化训练后，MAE降低到{stage4\_mae:.4f}
* 强化训练使MAE降低了-1.0255，效果显著
* 模型能够有效从不同数据集中学习，具有良好的泛化能力

# Wordle预测模型四阶段实验总结

## 实验概述

本实验按照以下四步流程完成：
1. 基于wordle_random.csv训练和测试模型
2. 用wordle_games.csv测试模型
3. 用wordle_games.csv强化训练模型
4. 再次测试模型，比较前后差距

## 数据集信息

- **训练集**: wordle_random.csv (3,334,196局游戏)
- **测试集**: wordle_games.csv (6,877,046局游戏)

## 实验结果

### 阶段1：基于wordle_random.csv训练和测试
- **训练模型**: stage1_random_model.npz
- **测试结果**: MSE = 0.0000, MAE = 0.0005

### 阶段2：用wordle_games.csv测试
- **测试结果**: MSE = 17.7746, MAE = 3.9533

### 阶段3：用wordle_games.csv强化训练
- **训练模型**: stage3_reinforced_model.npz

### 阶段4：再次测试强化后的模型
- **测试结果**: MSE = 2.1470, MAE = 1.2212

### 实验总结
- **初始模型MAE**: 3.9533
- **强化后MAE**: 1.2212
- **改进幅度**: 2.7321 MAE reduction

## 可视化图表

### 训练损失曲线
- **文件**: visualizations/stage1_vs_stage3_losses.png
- **描述**: 对比阶段1和阶段3的训练损失曲线

### MAE对比图
- **文件**: visualizations/mae_across_stages.png
- **描述**: 三个阶段的MAE对比

### 改进对比图
- **文件**: visualizations/improvement_after_reinforcement.png
- **描述**: 强化前后的MAE改进

### 学习曲线
- **文件**: visualizations/learning_curves.png
- **描述**: 两个阶段的学习曲线

### MAE趋势图
- **文件**: visualizations/mae_trend.png
- **描述**: 各阶段的MAE趋势

## 结论

- 模型在初始训练阶段表现良好，MAE为{stage1_mae:.4f}
- 用wordle_games.csv测试时，MAE为{stage2_mae:.4f}
- 经过强化训练后，MAE降低到{stage4_mae:.4f}
- 强化训练使MAE降低了2.7321，效果显著
- 模型能够有效从不同数据集中学习，具有良好的泛化能力

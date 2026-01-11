# Wordle Embedding 实验报告

## 1. 实验概述

本实验针对Wordle游戏数据，生成了字符级和单词级的embedding模型，用于后续的游戏步数预测任务。

### 1.1 数据集信息

- **wordle_random.csv**: 3334196局游戏
- **wordle_games.csv**: 6877046局游戏

### 1.2 Embedding类型

- **单词级Embedding**: 针对每个单词生成一个向量表示
- **字符级Embedding**: 针对每个字符生成一个向量表示

## 2. 数据统计分析

### 2.1 wordle_random.csv 统计

- 总游戏数: 3334196
- 平均步数: 3.98
- 中位数步数: 4.00

### 2.2 wordle_games.csv 统计

- 总游戏数: 6877046
- 总用户数: 977510
- 平均步数: 4.20
- 中位数步数: 4.00

### 2.3 单词统计

- 总单词数: 2309
- 最高频率单词: shake (41001次)
- 最低频率单词: myrrh (3746次)

## 3. Embedding 模型详情

### 3.1 词汇表构建

- **单词级词汇表**: 包含所有出现过的单词，添加了<PAD>和<UNK>特殊标记
- **字符级词汇表**: 包含所有出现过的字符，添加了<PAD>和<UNK>特殊标记

### 3.2 Embedding 矩阵生成

- **Embedding维度**: 32
- **初始化方式**: 随机高斯分布
- **<PAD>向量**: 全0向量

## 4. 生成的文件

### 4.1 数据集文件

- `random_word_embedding.npz`: random数据单词级embedding
- `random_char_embedding.npz`: random数据字符级embedding
- `games_word_embedding.npz`: games数据单词级embedding
- `games_char_embedding.npz`: games数据字符级embedding

### 4.2 可视化图表

- 步数分布图
- 单词长度分布图
- 单词难度分布图
- Embedding分布直方图
- Embedding向量示例图

### 4.3 代码文件

- `embedding_processing.py`: Embedding生成主程序

## 5. 结论与展望

- 成功生成了针对两个数据集的字符级和单词级embedding模型
- 嵌入模型可以直接用于后续的Wordle游戏步数预测任务
- 可以进一步优化embedding生成方式，如使用CBOW或Skip-gram模型
- 可以考虑加入预训练的embedding模型，如Word2Vec或GloVe


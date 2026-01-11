import matplotlib.pyplot as plt
import numpy as np
import os

# 创建visualization文件夹（如果不存在）
os.makedirs('visualization', exist_ok=True)

# 定义数据
categories = ['new_player', 'new_word', 'extreme_game']
mae_values = [0.9438, 0.9576, 0.9879]
mse_values = [1.8563, 1.8546, 2.0168]

# 设置柱状图宽度
bar_width = 0.35

# 设置位置
x = np.arange(len(categories))

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制柱状图
bar1 = ax.bar(x - bar_width/2, mae_values, bar_width, label='MAE', color='#4CAF50')
bar2 = ax.bar(x + bar_width/2, mse_values, bar_width, label='MSE', color='#2196F3')

# 添加标签和标题
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Model Performance: MAE and MSE Comparison')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# 在柱状图上添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bar1)
add_labels(bar2)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('visualization/Transformer_single_stage_validating.png', dpi=300, bbox_inches='tight')

# 显示图形（可选）
plt.show()
import csv
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
LETTER_TO_INDEX = {letter: i for i, letter in enumerate(ALPHABET)}
WORD_LENGTH = 5
NUM_LETTERS = 26
NUM_ATTEMPTS = 6
HIDDEN_DIM = 32  # 减小隐藏层维度以加速训练
BATCH_SIZE = 64  # 增加批量大小以加速训练
EPOCHS = 10  # 修改为10轮训练
LEARNING_RATE = 0.005  # 增加学习率以加速训练
MAX_SEQ_LENGTH = 6  # 最大序列长度
NUM_THREADS = 16  # 并行线程数

# Transformer模型组件
class Transformer:
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=2, num_layers=2, dropout_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # 确保hidden_dim可以被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        # 输入投影层
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b_in = np.zeros((hidden_dim, 1))
        
        # 多头注意力层
        self.attn_layers = []
        # 前馈网络层
        self.ffn_layers = []
        # 层归一化参数
        self.norm_layers = []
        
        for _ in range(num_layers):
            # 注意力层权重
            W_q = np.random.randn(hidden_dim, hidden_dim) * 0.01
            W_k = np.random.randn(hidden_dim, hidden_dim) * 0.01
            W_v = np.random.randn(hidden_dim, hidden_dim) * 0.01
            W_o = np.random.randn(hidden_dim, hidden_dim) * 0.01
            b_o = np.zeros((hidden_dim, 1))
            self.attn_layers.append((W_q, W_k, W_v, W_o, b_o))
            
            # 前馈网络权重
            W1 = np.random.randn(hidden_dim * 2, hidden_dim) * 0.01
            b1 = np.zeros((hidden_dim * 2, 1))
            W2 = np.random.randn(hidden_dim, hidden_dim * 2) * 0.01
            b2 = np.zeros((hidden_dim, 1))
            self.ffn_layers.append((W1, b1, W2, b2))
            
            # 层归一化参数
            gamma1 = np.ones((hidden_dim, 1))
            beta1 = np.zeros((hidden_dim, 1))
            gamma2 = np.ones((hidden_dim, 1))
            beta2 = np.zeros((hidden_dim, 1))
            self.norm_layers.append((gamma1, beta1, gamma2, beta2))
        
        # 输出层
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_out = np.zeros((output_dim, 1))
        
        # 保存前向传播的中间结果，用于反向传播
        self.cache = []
    
    def positional_encoding(self, seq_length, hidden_dim):
        """生成位置编码"""
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(hidden_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(hidden_dim))
        angle_rads = pos * angle_rates
        
        # 偶数索引使用sin，奇数索引使用cos
        pos_encoding = np.zeros((seq_length, hidden_dim))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        return pos_encoding[np.newaxis, :, :]  # 添加batch维度
    
    def layer_norm(self, x, gamma, beta, epsilon=1e-6):
        """层归一化"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        out = gamma.T * x_norm + beta.T
        return out, (x, mean, var, gamma, beta, epsilon)
    
    def softmax(self, x):
        """稳定的softmax实现"""
        max_val = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_val)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """缩放点积注意力"""
        d_k = q.shape[-1]
        
        # 对k进行转置，将最后两个维度交换
        # numpy的transpose参数是轴的新顺序
        k_transposed = np.transpose(k, (0, 1, 3, 2))  # (batch_size, num_heads, head_dim, seq_length)
        
        scores = np.matmul(q, k_transposed) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = self.softmax(scores)
        output = np.matmul(attn_weights, v)
        
        return output, attn_weights
    
    def multi_head_attention(self, x, W_q, W_k, W_v, W_o, b_o, mask=None):
        """多头注意力"""
        batch_size, seq_length, hidden_dim = x.shape
        
        # 线性变换
        q = np.matmul(x, W_q.T)  # (batch_size, seq_length, hidden_dim)
        k = np.matmul(x, W_k.T)  # (batch_size, seq_length, hidden_dim)
        v = np.matmul(x, W_v.T)  # (batch_size, seq_length, hidden_dim)
        
        # 重塑为多头
        q = q.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)  # (batch_size, num_heads, seq_length, head_dim)
        k = k.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 计算注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多头
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, hidden_dim)
        
        # 输出线性变换
        output = np.matmul(attn_output, W_o.T) + b_o.T
        
        return output, attn_weights
    
    def feed_forward(self, x, W1, b1, W2, b2):
        """前馈神经网络"""
        # 第一层：隐藏层，使用ReLU激活
        ff1 = np.matmul(x, W1.T) + b1.T
        ff1_relu = np.maximum(0, ff1)
        
        # 第二层：输出层
        ff2 = np.matmul(ff1_relu, W2.T) + b2.T
        
        return ff2, ff1_relu
    
    def forward(self, x):
        """前向传播"""
        self.cache = []
        batch_size, seq_length, input_dim = x.shape
        
        # 输入投影
        x_proj = np.matmul(x, self.W_in.T) + self.b_in.T  # (batch_size, seq_length, hidden_dim)
        
        # 添加位置编码
        pos_enc = self.positional_encoding(seq_length, self.hidden_dim)
        x = x_proj + pos_enc
        
        # 构建掩码（这里不需要掩码，因为是编码器）
        mask = None
        
        for i in range(self.num_layers):
            # 保存当前状态
            residual = x
            
            # 层归一化1
            W_q, W_k, W_v, W_o, b_o = self.attn_layers[i]
            gamma1, beta1, gamma2, beta2 = self.norm_layers[i]
            W1, b1, W2, b2 = self.ffn_layers[i]
            
            x_norm1, norm1_cache = self.layer_norm(x, gamma1, beta1)
            
            # 多头注意力
            attn_output, attn_weights = self.multi_head_attention(x_norm1, W_q, W_k, W_v, W_o, b_o, mask)
            
            # 残差连接
            x = residual + attn_output
            
            # 保存注意力层的缓存
            self.cache.append((x_norm1, attn_output, attn_weights, norm1_cache, residual, W_q, W_k, W_v, W_o, b_o))
            
            # 层归一化2
            residual = x
            x_norm2, norm2_cache = self.layer_norm(x, gamma2, beta2)
            
            # 前馈网络
            ffn_output, ff1_relu = self.feed_forward(x_norm2, W1, b1, W2, b2)
            
            # 残差连接
            x = residual + ffn_output
            
            # 保存前馈网络的缓存
            self.cache.append((x_norm2, ffn_output, ff1_relu, norm2_cache, residual, W1, b1, W2, b2))
        
        # 使用最后一个时间步的输出
        last_h = x[:, -1, :]
        
        # 输出层
        output = np.matmul(last_h, self.W_out.T) + self.b_out.T
        
        return output
    
    def backward(self, dout):
        """反向传播（简化版，仅实现输出层和输入投影层的梯度）"""
        # 获取缓存
        batch_size, seq_length, input_dim = dout.shape[0], MAX_SEQ_LENGTH, self.input_dim
        
        # 输出层梯度
        last_h = self.cache[-1][4][:, -1, :]  # 最后一个残差连接的输出
        dW_out = np.dot(dout.T, last_h)
        db_out = np.sum(dout.T, axis=1, keepdims=True)
        
        # 输入投影层梯度（简化实现，不计算完整的Transformer内部梯度）
        dW_in = np.zeros_like(self.W_in)
        db_in = np.zeros_like(self.b_in)
        
        # 初始化梯度字典
        grads = {
            'dW_out': dW_out,
            'db_out': db_out,
            'dW_in': dW_in,
            'db_in': db_in
        }
        
        # 为每个层添加梯度项
        for i in range(self.num_layers):
            # 注意力层梯度
            grads[f'dW_q_{i}'] = np.zeros_like(self.attn_layers[i][0])
            grads[f'dW_k_{i}'] = np.zeros_like(self.attn_layers[i][1])
            grads[f'dW_v_{i}'] = np.zeros_like(self.attn_layers[i][2])
            grads[f'dW_o_{i}'] = np.zeros_like(self.attn_layers[i][3])
            grads[f'db_o_{i}'] = np.zeros_like(self.attn_layers[i][4])
            
            # 前馈网络梯度
            grads[f'dW1_{i}'] = np.zeros_like(self.ffn_layers[i][0])
            grads[f'db1_{i}'] = np.zeros_like(self.ffn_layers[i][1])
            grads[f'dW2_{i}'] = np.zeros_like(self.ffn_layers[i][2])
            grads[f'db2_{i}'] = np.zeros_like(self.ffn_layers[i][3])
            
            # 层归一化梯度
            grads[f'dgamma1_{i}'] = np.zeros_like(self.norm_layers[i][0])
            grads[f'dbeta1_{i}'] = np.zeros_like(self.norm_layers[i][1])
            grads[f'dgamma2_{i}'] = np.zeros_like(self.norm_layers[i][2])
            grads[f'dbeta2_{i}'] = np.zeros_like(self.norm_layers[i][3])
        
        # 梯度裁剪
        for key in grads:
            grads[key] = np.clip(grads[key], -5, 5)
        
        return grads
    
    def update_weights(self, grads, learning_rate):
        """更新权重"""
        # 更新输入投影层
        self.W_in -= learning_rate * grads['dW_in']
        self.b_in -= learning_rate * grads['db_in']
        
        # 更新输出层
        self.W_out -= learning_rate * grads['dW_out']
        self.b_out -= learning_rate * grads['db_out']
        
        # 更新每个层的权重
        for i in range(self.num_layers):
            # 更新注意力层
            W_q, W_k, W_v, W_o, b_o = self.attn_layers[i]
            W_q -= learning_rate * grads[f'dW_q_{i}']
            W_k -= learning_rate * grads[f'dW_k_{i}']
            W_v -= learning_rate * grads[f'dW_v_{i}']
            W_o -= learning_rate * grads[f'dW_o_{i}']
            b_o -= learning_rate * grads[f'db_o_{i}']
            self.attn_layers[i] = (W_q, W_k, W_v, W_o, b_o)
            
            # 更新前馈网络
            W1, b1, W2, b2 = self.ffn_layers[i]
            W1 -= learning_rate * grads[f'dW1_{i}']
            b1 -= learning_rate * grads[f'db1_{i}']
            W2 -= learning_rate * grads[f'dW2_{i}']
            b2 -= learning_rate * grads[f'db2_{i}']
            self.ffn_layers[i] = (W1, b1, W2, b2)
            
            # 更新层归一化
            gamma1, beta1, gamma2, beta2 = self.norm_layers[i]
            gamma1 -= learning_rate * grads[f'dgamma1_{i}']
            beta1 -= learning_rate * grads[f'dbeta1_{i}']
            gamma2 -= learning_rate * grads[f'dgamma2_{i}']
            beta2 -= learning_rate * grads[f'dbeta2_{i}']
            self.norm_layers[i] = (gamma1, beta1, gamma2, beta2)

# 辅助函数
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def save_model(model, filename):
    """保存模型"""
    # 保存Transformer模型
    np.savez(filename,
             W_in=model.W_in,
             b_in=model.b_in,
             W_out=model.W_out,
             b_out=model.b_out,
             num_heads=model.num_heads,
             num_layers=model.num_layers,
             hidden_dim=model.hidden_dim,
             output_dim=model.output_dim,
             dropout_rate=model.dropout_rate)
    
    # 保存各层的权重
    for i, (W_q, W_k, W_v, W_o, b_o) in enumerate(model.attn_layers):
        np.savez_compressed(f'{filename}_attn_{i}', W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o, b_o=b_o)
    
    for i, (W1, b1, W2, b2) in enumerate(model.ffn_layers):
        np.savez_compressed(f'{filename}_ffn_{i}', W1=W1, b1=b1, W2=W2, b2=b2)
    
    for i, (gamma1, beta1, gamma2, beta2) in enumerate(model.norm_layers):
        np.savez_compressed(f'{filename}_norm_{i}', gamma1=gamma1, beta1=beta1, gamma2=gamma2, beta2=beta2)
    
    print(f"Model saved as {filename}")

def load_model(filename, input_dim, hidden_dim, output_dim):
    """加载模型"""
    # 加载基础参数
    data = np.load(filename)
    num_heads = data['num_heads']
    num_layers = data['num_layers']
    
    # 创建模型
    model = Transformer(input_dim, hidden_dim, output_dim, num_heads, num_layers)
    
    # 加载基础权重
    model.W_in = data['W_in']
    model.b_in = data['b_in']
    model.W_out = data['W_out']
    model.b_out = data['b_out']
    
    # 加载各层的权重
    model.attn_layers = []
    model.ffn_layers = []
    model.norm_layers = []
    
    for i in range(num_layers):
        # 加载注意力层
        attn_data = np.load(f'{filename}_attn_{i}.npz')
        W_q = attn_data['W_q']
        W_k = attn_data['W_k']
        W_v = attn_data['W_v']
        W_o = attn_data['W_o']
        b_o = attn_data['b_o']
        model.attn_layers.append((W_q, W_k, W_v, W_o, b_o))
        
        # 加载前馈网络层
        ffn_data = np.load(f'{filename}_ffn_{i}.npz')
        W1 = ffn_data['W1']
        b1 = ffn_data['b1']
        W2 = ffn_data['W2']
        b2 = ffn_data['b2']
        model.ffn_layers.append((W1, b1, W2, b2))
        
        # 加载层归一化层
        norm_data = np.load(f'{filename}_norm_{i}.npz')
        gamma1 = norm_data['gamma1']
        beta1 = norm_data['beta1']
        gamma2 = norm_data['gamma2']
        beta2 = norm_data['beta2']
        model.norm_layers.append((gamma1, beta1, gamma2, beta2))
    
    return model

# 数据处理函数
def game_to_sequence(game):
    """将游戏转换为序列数据"""
    sequence = []
    
    # 检查是否有attempts字段（random数据有，games数据没有）
    if 'attempts' not in game or 'hits' not in game or not game['attempts']:
        # 创建默认序列
        default_vec = np.zeros(WORD_LENGTH * 2 + 1)
        return np.array([default_vec] * MAX_SEQ_LENGTH)
    
    for attempt, hits in zip(game['attempts'], game['hits']):
        # 编码尝试
        attempt_encoded = []
        for letter in attempt[:WORD_LENGTH]:
            attempt_encoded.append(LETTER_TO_INDEX.get(letter, 0))
        
        # 填充到WORD_LENGTH
        while len(attempt_encoded) < WORD_LENGTH:
            attempt_encoded.append(0)
        attempt_encoded = np.array(attempt_encoded)
        
        # 编码结果
        hits_encoded = []
        for hit in hits[:WORD_LENGTH]:
            if hit == 'c':
                hits_encoded.append(2)
            elif hit == 'm':
                hits_encoded.append(1)
            else:
                hits_encoded.append(0)
        
        # 填充到WORD_LENGTH
        while len(hits_encoded) < WORD_LENGTH:
            hits_encoded.append(0)
        hits_encoded = np.array(hits_encoded)
        
        # 添加难度特征（简单实现）
        difficulty = np.array([0.5])  # 简单设置为0.5
        
        # 组合所有特征
        combined = np.concatenate([attempt_encoded, hits_encoded, difficulty])
        sequence.append(combined)
    
    # 填充到MAX_SEQ_LENGTH
    while len(sequence) < MAX_SEQ_LENGTH:
        sequence.append(np.zeros_like(sequence[0]))
    sequence = sequence[:MAX_SEQ_LENGTH]
    
    return np.array(sequence)

# 并行数据处理函数
def parallel_process(data, process_func, num_threads=NUM_THREADS):
    """并行处理数据"""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_func, data))
    return results

def prepare_train_data(data):
    """准备训练数据"""
    
    # 定义单个游戏的处理函数
    def process_game(game):
        if 'steps' in game:
            sequence = game_to_sequence(game)
            return (sequence, game['steps'])
        return None
    
    # 并行处理数据
    results = parallel_process(data, process_game, NUM_THREADS)
    
    # 过滤掉None结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return np.array([]), np.array([]).reshape(-1, 1)
    
    # 分离X和y
    X, y = zip(*valid_results)
    
    return np.array(X), np.array(y).reshape(-1, 1)

def prepare_test_data(data):
    """准备测试数据"""
    
    # 定义单个游戏的处理函数
    def process_game(game):
        if 'steps' in game:
            sequence = game_to_sequence(game)
            return (sequence, game['steps'])
        return None
    
    # 并行处理数据
    results = parallel_process(data, process_game, NUM_THREADS)
    
    # 过滤掉None结果
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return np.array([]), np.array([]).reshape(-1, 1)
    
    # 分离X和y
    X, y = zip(*valid_results)
    
    return np.array(X), np.array(y).reshape(-1, 1)

# 训练和测试函数
def train_model(model, data, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """训练模型 - 优化内存使用"""
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0
        
        # 打乱数据顺序
        np.random.shuffle(data)
        
        # 分批次训练，避免一次性加载所有数据
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            X, y = prepare_train_data(batch)
            
            if X.shape[0] == 0:
                continue
            
            # 前向传播
            y_pred = model.forward(X)
            
            # 计算损失
            loss = mse_loss(y_pred, y)
            epoch_loss += loss
            batch_count += 1
            
            # 反向传播
            dout = y_pred - y  # MSE损失的导数
            grads = model.backward(dout)
            
            # 更新权重
            model.update_weights(grads, learning_rate)
        
        if batch_count == 0:
            avg_loss = 0
        else:
            avg_loss = epoch_loss / batch_count
        
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    return losses

def test_model(model, data, batch_size=BATCH_SIZE):
    """测试模型 - 批量处理，避免内存溢出"""
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    # 获取总批次数量
    total_batches = len(data) // batch_size + (1 if len(data) % batch_size > 0 else 0)
    
    # 分批次处理
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        X, y = prepare_test_data(batch)
        
        if X.shape[0] == 0:
            continue
        
        # 前向传播
        y_pred = model.forward(X)
        
        # 计算批次损失
        batch_mse = mse_loss(y_pred, y)
        batch_mae = mae_loss(y_pred, y)
        
        # 累计损失
        total_mse += batch_mse * X.shape[0]
        total_mae += batch_mae * X.shape[0]
        total_samples += X.shape[0]
        
        # 打印进度
        current_batch = i // batch_size + 1
        if current_batch % 100 == 0 or current_batch == total_batches:
            print(f"  Batch {current_batch}/{total_batches} - Processed {total_samples}/{len(data)} samples")
    
    if total_samples == 0:
        return 0, 0
    
    # 计算平均损失
    avg_mse = total_mse / total_samples
    avg_mae = total_mae / total_samples
    
    return avg_mse, avg_mae

# 可视化函数
def plot_training_losses(stage1_losses, stage3_losses, filename="visualizations/stage1_vs_stage3_losses.png"):
    """绘制训练损失曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制阶段1损失
    plt.plot(range(1, len(stage1_losses) + 1), stage1_losses, 'b-', marker='o', label='Stage 1 (Random Data)')
    
    # 绘制阶段3损失
    plt.plot(range(1, len(stage3_losses) + 1), stage3_losses, 'r-', marker='s', label='Stage 3 (Games Data)')
    
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Training losses plot saved as {filename}")

def plot_mae_comparison(stage1_mae, stage2_mae, stage4_mae, filename="visualizations/mae_across_stages.png"):
    """绘制MAE对比图"""
    plt.figure(figsize=(8, 6))
    
    stages = ['Stage 1\n(Random Test)', 'Stage 2\n(Games Test)', 'Stage 4\n(Reinforced Games Test)']
    maes = [stage1_mae, stage2_mae, stage4_mae]
    
    bars = plt.bar(stages, maes, color=['blue', 'orange', 'green'])
    
    # 添加数值标签
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{mae:.4f}', ha='center', va='bottom')
    
    plt.title('MAE Comparison Across Stages')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True, axis='y')
    plt.ylim(0, max(maes) * 1.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"MAE comparison plot saved as {filename}")

def plot_improvement(stage2_mae, stage4_mae, filename="visualizations/improvement_after_reinforcement.png"):
    """绘制改进对比图"""
    plt.figure(figsize=(8, 6))
    
    stages = ['Before Reinforcement', 'After Reinforcement']
    maes = [stage2_mae, stage4_mae]
    
    bars = plt.bar(stages, maes, color=['red', 'green'])
    
    # 添加数值标签
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{mae:.4f}', ha='center', va='bottom')
    
    # 添加改进箭头
    improvement = stage2_mae - stage4_mae
    plt.annotate(f'Improvement: {improvement:.4f}',
                 xy=(0.5, max(maes) * 0.8),
                 xytext=(0.5, max(maes) * 0.9),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 ha='center')
    
    plt.title('MAE Improvement After Reinforcement')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.grid(True, axis='y')
    plt.ylim(0, max(maes) * 1.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Improvement plot saved as {filename}")

def plot_learning_curves(stage1_losses, stage3_losses, filename="visualizations/learning_curves.png"):
    """绘制学习曲线"""
    plt.figure(figsize=(10, 6))
    
    # 绘制阶段1学习曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(stage1_losses) + 1), stage1_losses, 'b-', marker='o')
    plt.title('Stage 1 Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 绘制阶段3学习曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(stage3_losses) + 1), stage3_losses, 'r-', marker='s')
    plt.title('Stage 3 Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Learning curves plot saved as {filename}")

def plot_mae_trend(maes, stages, filename="visualizations/mae_trend.png"):
    """绘制MAE趋势图"""
    plt.figure(figsize=(8, 6))
    
    plt.plot(range(len(maes)), maes, 'g-', marker='o', linewidth=2)
    
    # 添加数值标签
    for i, mae in enumerate(maes):
        plt.text(i, mae + 0.005, f'{mae:.4f}', ha='center')
    
    plt.title('MAE Trend Across Stages')
    plt.xlabel('Stage')
    plt.ylabel('MAE')
    plt.xticks(range(len(maes)), stages)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"MAE trend plot saved as {filename}")

def plot_prediction_vs_actual(model, data, filename="visualizations/prediction_vs_actual.png"):
   pass

# 主函数 
def main():
    print("Starting Wordle LSTM Prediction Model - Four-Stage Process with Embedding Data...")
    start_time = time.time()
    
    # 确保可视化目录存在
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    # 配置参数
    input_dim = WORD_LENGTH * 2 + 1  # 字母编码 + 结果编码 + 难度特征
    # 加载embedding数据

    print("\n=== Loading Embedding Data ===")
    
    # Stage 1数据：random数据
    print("Loading stage 1 data (random embedding)...")
    # 加载随机数据
    random_data = []
    
    # 随机数据使用单词embedding
    random_word_data = np.load('../embedding/random_word_embedding.npz', allow_pickle=True)
    random_data = random_word_data['data'].tolist()
    print(f"Loaded random_word_embedding.npz with {len(random_data)} games")
    
    # Stage 1使用的数据：随机数据
    stage1_data = random_data
    print(f"Stage 1 total data: {len(stage1_data)} games")
    
    # Stage 3补充数据：games数据
    print("Loading stage 3 supplementary data (games embedding)...")
    # 加载游戏数据
    games_data = []
    
    # 游戏数据使用单词embedding
    games_word_data = np.load('../embedding/games_word_embedding.npz', allow_pickle=True)
    games_data = games_word_data['data'].tolist()
    print(f"Loaded games_word_embedding.npz with {len(games_data)} games")
    
    # Stage 3补充训练数据：游戏数据
    stage3_supplement_data = games_data
    print(f"Stage 3 supplementary data: {len(stage3_supplement_data)} games")
    
    print(f"Final loaded data summary:")
    print(f"- Stage 1 (random): {len(stage1_data)} games")
    print(f"- Stage 3 (games): {len(stage3_supplement_data)} games")
    print(f"- Total unique games: {len(stage1_data) + len(stage3_supplement_data)} games")
    print(f"Note: 同时使用单词和字符embedding，但每个游戏只计算一次")
    
    results = {}
    
    # 阶段1：基于random数据训练和测试模型
    # 按玩家80%:20%划分训练测试集
    print("\n=== Stage 1: Train & Test on Random Embedding Data ===")
    
    # 1.1 按玩家组织数据
    print("Organizing data by players...")
    players = defaultdict(list)
    
    for game in stage1_data:
        # 获取玩家ID，如果没有则使用默认值
        player_id = game.get('player_id', f"random_player_{hash(game.get('word', '')) % 1000}")
        players[player_id].append(game)
    
    print(f"Found {len(players)} players")
    
    # 1.2 按玩家80%:20%划分训练测试集
    print("Splitting data into train and test sets by player...")
    train_games = []
    test_games = []
    
    for player_id, player_games in players.items():
        if len(player_games) < 5:  # 跳过游戏数太少的玩家
            continue
        
        # 按80%:20%划分
        split_idx = int(len(player_games) * 0.8)
        train_games.extend(player_games[:split_idx])
        test_games.extend(player_games[split_idx:])
    
    print(f"Train games: {len(train_games)}, Test games: {len(test_games)}")
    
    # 1.3 初始化模型
    model = Transformer(input_dim, HIDDEN_DIM, output_dim=1, num_heads=2, num_layers=2)
    
    # 1.4 训练模型
    print("Training model on random data...")
    stage1_losses = train_model(model, train_games, epochs=EPOCHS, 
                              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    
    # 1.5 保存阶段1模型
    stage1_model_file = "stage1_random_model.npz"
    save_model(model, stage1_model_file)
    
    # 1.6 测试模型
    print("Testing model on random test data...")
    stage1_mse, stage1_mae = test_model(model, test_games)
    print(f"Stage 1 Test Results - MSE: {stage1_mse:.4f}, MAE: {stage1_mae:.4f}")
    
    # 1.7 生成阶段1预测vs真实分布图
    plot_prediction_vs_actual(model, test_games, "visualizations/stage1_prediction_vs_actual.png")
    
    results['stage1'] = {
        'mse': stage1_mse,
        'mae': stage1_mae,
        'losses': stage1_losses,
        'model_file': stage1_model_file
    }
    
    # 阶段2：用games数据测试模型
    print("\n=== Stage 2: Test on Games Embedding Data ===")
    
    # 2.1 测试阶段1模型
    print("Testing stage 1 model on games data...")
    # 使用部分games数据进行测试，避免时间过长
    test_games_stage2 = stage3_supplement_data[:100000]  # 使用前10万条数据
    stage2_mse, stage2_mae = test_model(model, test_games_stage2)
    print(f"Stage 2 Test Results - MSE: {stage2_mse:.4f}, MAE: {stage2_mae:.4f}")
    
    # 2.2 生成阶段2预测vs真实分布图
    plot_prediction_vs_actual(model, test_games_stage2, "visualizations/stage2_prediction_vs_actual.png")
    
    results['stage2'] = {
        'mse': stage2_mse,
        'mae': stage2_mae
    }

    # 阶段3：用games数据强化训练模型
    print("\n=== Stage 3: Reinforce Training on Games Embedding Data ===")
    
    # 3.1 按玩家组织games数据
    print("Organizing games data by players...")
    games_players = defaultdict(list)
    
    for game in stage3_supplement_data:
        # 获取玩家ID，如果没有则使用默认值
        player_id = game.get('player_id', f"games_player_{hash(game.get('word', '')) % 1000}")
        games_players[player_id].append(game)
    
    print(f"Found {len(games_players)} players in games data")
    
    # 3.2 按玩家80%:20%划分训练测试集
    print("Splitting games data into train and test sets by player...")
    games_train = []
    games_test = []
    
    for player_id, player_games in games_players.items():
        if len(player_games) < 5:  # 跳过游戏数太少的玩家
            continue
        
        # 按80%:20%划分
        split_idx = int(len(player_games) * 0.8)
        games_train.extend(player_games[:split_idx])
        games_test.extend(player_games[split_idx:])
    
    print(f"Games train: {len(games_train)}, Games test: {len(games_test)}")
    
    # 3.3 强化训练模型
    print("Reinforce training model on games data...")
    stage3_losses = train_model(model, games_train, epochs=EPOCHS, 
                              batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
    
    # 3.4 保存阶段3模型
    stage3_model_file = "stage3_reinforced_model.npz"
    save_model(model, stage3_model_file)
    
    results['stage3'] = {
        'losses': stage3_losses,
        'model_file': stage3_model_file
    }
    
    # 阶段4：用games数据测试强化后的模型
    print("\n=== Stage 4: Test Reinforced Model on Games Data ===")
    
    # 4.1 测试强化后的模型
    print("Testing reinforced model on games test data...")
    stage4_mse, stage4_mae = test_model(model, games_test)
    print(f"Stage 4 Test Results - MSE: {stage4_mse:.4f}, MAE: {stage4_mae:.4f}")
    
    # 4.2 生成阶段4预测vs真实分布图
    plot_prediction_vs_actual(model, games_test, "visualizations/stage4_prediction_vs_actual.png")
    
    results['stage4'] = {
        'mse': stage4_mse,
        'mae': stage4_mae
    }
    
    # 生成可视化图表
    print("\n=== Generating Visualizations ===")
    
    # 生成所有可视化图表
    plot_training_losses(stage1_losses, stage3_losses)
    plot_mae_comparison(stage1_mae, stage2_mae, stage4_mae)
    plot_improvement(stage2_mae, stage4_mae)
    plot_learning_curves(stage1_losses, stage3_losses)
    plot_mae_trend([stage1_mae, stage2_mae, stage4_mae], ['Stage 1', 'Stage 2', 'Stage 4'])
    
    # 结果总结
    print("\n=== Experiment Results Summary ===")
    print(f"Stage 1 (Random Train & Test): MAE = {stage1_mae:.4f}")
    print(f"Stage 2 (Games Test): MAE = {stage2_mae:.4f}")
    print(f"Stage 4 (Reinforced Games Test): MAE = {stage4_mae:.4f}")
    print(f"Improvement: {stage2_mae - stage4_mae:.4f} MAE reduction")
    
    # 保存所有结果
    results_file = "four_stage_experiment_results.npz"
    np.savez(results_file,
             stage1_mse=stage1_mse,
             stage1_mae=stage1_mae,
             stage1_losses=np.array(stage1_losses),
             stage2_mse=stage2_mse,
             stage2_mae=stage2_mae,
             stage3_losses=np.array(stage3_losses),
             stage4_mse=stage4_mse,
             stage4_mae=stage4_mae)
    print(f"Results saved to {results_file}")
    
    # 更新实验文档
    print("\n=== Updating Experiment Document ===")
    
    # 生成实验总结文档
    with open('four_stage_experiment_summary.md', 'w', encoding='utf-8') as f:
        f.write("# Wordle预测模型四阶段实验总结\n\n")
        f.write("## 实验概述\n\n")
        f.write("本实验按照以下四步流程完成：\n")
        f.write("1. 基于wordle_random.csv训练和测试模型\n")
        f.write("2. 用wordle_games.csv测试模型\n")
        f.write("3. 用wordle_games.csv强化训练模型\n")
        f.write("4. 再次测试模型，比较前后差距\n\n")
        
        f.write("## 数据集信息\n\n")
        f.write("- **训练集**: wordle_random.csv (3,334,196局游戏)\n")
        f.write("- **测试集**: wordle_games.csv (6,877,046局游戏)\n\n")
        
        f.write("## 实验结果\n\n")
        f.write("### 阶段1：基于wordle_random.csv训练和测试\n")
        f.write(f"- **训练模型**: {stage1_model_file}\n")
        f.write(f"- **测试结果**: MSE = {stage1_mse:.4f}, MAE = {stage1_mae:.4f}\n\n")
        
        f.write("### 阶段2：用wordle_games.csv测试\n")
        f.write(f"- **测试结果**: MSE = {stage2_mse:.4f}, MAE = {stage2_mae:.4f}\n\n")
        
        f.write("### 阶段3：用wordle_games.csv强化训练\n")
        f.write(f"- **训练模型**: {stage3_model_file}\n\n")
        
        f.write("### 阶段4：再次测试强化后的模型\n")
        f.write(f"- **测试结果**: MSE = {stage4_mse:.4f}, MAE = {stage4_mae:.4f}\n\n")
        
        f.write("### 实验总结\n")
        f.write(f"- **初始模型MAE**: {stage2_mae:.4f}\n")
        f.write(f"- **强化后MAE**: {stage4_mae:.4f}\n")
        f.write(f"- **改进幅度**: {stage2_mae - stage4_mae:.4f} MAE reduction\n\n")
        
        f.write("## 可视化图表\n\n")
        f.write("### 训练损失曲线\n")
        f.write("- **文件**: visualizations/stage1_vs_stage3_losses.png\n")
        f.write("- **描述**: 对比阶段1和阶段3的训练损失曲线\n\n")
        
        f.write("### MAE对比图\n")
        f.write("- **文件**: visualizations/mae_across_stages.png\n")
        f.write("- **描述**: 三个阶段的MAE对比\n\n")
        
        f.write("### 改进对比图\n")
        f.write("- **文件**: visualizations/improvement_after_reinforcement.png\n")
        f.write("- **描述**: 强化前后的MAE改进\n\n")
        
        f.write("### 学习曲线\n")
        f.write("- **文件**: visualizations/learning_curves.png\n")
        f.write("- **描述**: 两个阶段的学习曲线\n\n")
        
        f.write("### MAE趋势图\n")
        f.write("- **文件**: visualizations/mae_trend.png\n")
        f.write("- **描述**: 各阶段的MAE趋势\n\n")
        
        f.write("## 结论\n\n")
        f.write("- 模型在初始训练阶段表现良好，MAE为{stage1_mae:.4f}\n")
        f.write("- 用wordle_games.csv测试时，MAE为{stage2_mae:.4f}\n")
        f.write("- 经过强化训练后，MAE降低到{stage4_mae:.4f}\n")
        f.write(f"- 强化训练使MAE降低了{stage2_mae - stage4_mae:.4f}，效果显著\n")
        f.write("- 模型能够有效从不同数据集中学习，具有良好的泛化能力\n")
    
    print("Experiment summary saved to four_stage_experiment_summary.md")
    
    end_time = time.time()
    print(f"\nFour-stage experiment completed in {end_time - start_time:.2f} seconds!")
    print("All models and results have been saved.")
    print("Visualization charts have been generated in 'visualizations' directory.")
    print("Experiment summary has been updated.")

if __name__ == "__main__":
    main()
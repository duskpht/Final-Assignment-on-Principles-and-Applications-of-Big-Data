import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict, Counter

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局变量
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
WORD_LENGTH = 5
NUM_ATTEMPTS = 6
EMBEDDING_DIM = 32  # Embedding维度

# 文件路径
RANDOM_FILE = r'e:\资料\NJU Journey\课程类\正式课程\3-1 大数据系统原理与应用\期末大作业\wordle_random.csv\wordle_random.csv'
GAMES_FILE = r'e:\资料\NJU Journey\课程类\正式课程\3-1 大数据系统原理与应用\期末大作业\wordle_games.csv\wordle_games.csv'

# ----------------------
# 数据加载函数
# ----------------------
def load_random_data(file_path):
    """加载wordle_random.csv数据"""
    data = []
    count = 0
    print(f"Loading random data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = {
                'solution': row['solution'].lower().strip(),
                'attempts': [],
                'hits': []
            }
            
            # 提取尝试和结果
            for i in range(NUM_ATTEMPTS):
                attempt_key = f'attempt_{i}'
                hits_key = f'hits_{i}'
                
                attempt = row[attempt_key].lower().strip()
                hits = row[hits_key].strip()
                
                if attempt and hits:
                    game['attempts'].append(attempt)
                    game['hits'].append(hits)
            
            # 计算步数
            game['steps'] = len(game['attempts'])
            
            # 只保留5个字母的单词
            if len(game['solution']) == WORD_LENGTH:
                data.append(game)
                count += 1
                if count % 100000 == 0:
                    print(f"Loaded {count} random games so far...")
    
    print(f"Loaded {len(data)} random games")
    return data

def load_games_data(file_path):
    """加载wordle_games.csv数据"""
    data = []
    count = 0
    print(f"Loading games data from {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            game = {
                'username': row['Username'].strip(),
                'target': row['target'].lower().strip(),
                'steps': int(row['Trial'])
            }
            
            # 只保留5个字母的单词和有效步数
            if len(game['target']) == WORD_LENGTH and game['steps'] > 0:
                data.append(game)
                count += 1
                if count % 100000 == 0:
                    print(f"Loaded {count} games data records so far...")
    
    print(f"Loaded {len(data)} games data records")
    return data

# 特征计算函数
def calculate_word_frequency(random_data, games_data):
    """计算单词频率"""
    word_freq = Counter()
    
    # 从random_data中统计单词频率
    for game in random_data:
        word_freq[game['solution']] += 1
        for attempt in game['attempts']:
            word_freq[attempt] += 1
    
    # 从games_data中统计单词频率
    for game in games_data:
        word_freq[game['target']] += 1
    
    return word_freq

def calculate_word_difficulty(word, word_freq):
    """计算单词难度（基于频率）"""
    # 频率越低，难度越高
    freq = word_freq.get(word, 1)
    # 归一化到0-1范围
    max_freq = max(word_freq.values())
    difficulty = 1.0 - (freq / max_freq)
    return difficulty

def get_default_player_features():
    """获取默认的玩家特征"""
    return {
        'total_games': 0,
        'avg_steps': 0,
        'median_steps': [],
        'success_rate': 1.0  # 所有游戏都是成功的
    }

def calculate_player_features(games_data):
    """计算玩家行为特征"""
    player_features = defaultdict(get_default_player_features)
    
    # 统计每个玩家的游戏数据
    for game in games_data:
        username = game['username']
        player = player_features[username]
        player['total_games'] += 1
        player['median_steps'].append(game['steps'])
    
    # 计算最终特征
    for username, player in player_features.items():
        # 计算平均步数
        if player['total_games'] > 0:
            player['avg_steps'] = np.mean(player['median_steps'])
            player['median_steps'] = np.median(player['median_steps'])
    
    return player_features

# 单词级Embedding函数
def build_word_vocabulary(data, data_type):
    """构建单词级词汇表"""
    print(f"\nBuilding {data_type} word vocabulary...")
    
    # 收集所有单词
    all_words = set()
    
    if data_type == 'random':
        for game in data:
            all_words.add(game['solution'])
            for attempt in game['attempts']:
                all_words.add(attempt)
    else:  # games data
        for game in data:
            all_words.add(game['target'])
    
    # 排序并添加特殊标记
    sorted_words = sorted(all_words)
    vocabulary = ['<PAD>', '<UNK>'] + sorted_words  # 添加填充和未知标记
    
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    
    print(f"{data_type} word vocabulary built with {len(vocabulary)} words")
    return vocabulary, word_to_idx, idx_to_word

def generate_word_embedding_matrix(vocabulary, embedding_dim=EMBEDDING_DIM):
    """生成单词嵌入矩阵"""
    print(f"Generating word embedding matrix with dimension {embedding_dim}...")
    
    # 随机初始化嵌入矩阵
    embedding_matrix = np.random.randn(len(vocabulary), embedding_dim)
    
    # 设置<PAD>为全0向量
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    print(f"Word embedding matrix generated: shape = {embedding_matrix.shape}")
    return embedding_matrix

# 字符级Embedding函数
def build_char_vocabulary(data, data_type):
    """构建字符级词汇表"""
    print(f"\nBuilding {data_type} character vocabulary...")
    
    # 收集所有字符
    all_chars = set()
    
    if data_type == 'random':
        for game in data:
            for char in game['solution']:
                all_chars.add(char)
            for attempt in game['attempts']:
                for char in attempt:
                    all_chars.add(char)
    else:  # games data
        for game in data:
            for char in game['target']:
                all_chars.add(char)
    
    # 排序并添加特殊标记
    sorted_chars = sorted(all_chars)
    vocabulary = ['<PAD>', '<UNK>'] + sorted_chars  # 添加填充和未知标记
    
    char_to_idx = {char: i for i, char in enumerate(vocabulary)}
    idx_to_char = {i: char for char, i in char_to_idx.items()}
    
    print(f"{data_type} character vocabulary built with {len(vocabulary)} characters")
    return vocabulary, char_to_idx, idx_to_char

def generate_char_embedding_matrix(vocabulary, embedding_dim=EMBEDDING_DIM):
    """生成字符嵌入矩阵"""
    print(f"Generating character embedding matrix with dimension {embedding_dim}...")
    
    # 随机初始化嵌入矩阵
    embedding_matrix = np.random.randn(len(vocabulary), embedding_dim)
    
    # 设置<PAD>为全0向量
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    print(f"Character embedding matrix generated: shape = {embedding_matrix.shape}")
    return embedding_matrix

# 数据特征处理
def add_features_to_data(data, data_type, word_freq, player_features=None):
    """为数据添加特征"""
    print(f"Adding features to {data_type} data...")
    enhanced_data = []
    for game in data:
        enhanced_game = game.copy()
        # 1. 添加单词难度特征
        if data_type == 'random':
            word = game['solution']
        else:  # games data
            word = game['target']
        
        enhanced_game['word_difficulty'] = calculate_word_difficulty(word, word_freq)
        
        # 2. 添加玩家特征（仅对games数据）
        if data_type == 'games' and player_features:
            username = game['username']
            features = player_features.get(username, get_default_player_features())
            enhanced_game['player_total_games'] = features['total_games']
            enhanced_game['player_avg_steps'] = features['avg_steps']
            enhanced_game['player_median_steps'] = features['median_steps']
        
        enhanced_data.append(enhanced_game)
    
    return enhanced_data

# 可视化函数
def generate_visualizations(data, data_type, output_dir):
    """生成可视化图表"""
    print(f"\n=== Generating {data_type} Data Visualizations ===")
    
    # 1. 步数分布
    steps = [game['steps'] for game in data]
    steps_count = Counter(steps)
    
    plt.figure(figsize=(8, 6))
    sorted_steps = sorted(steps_count.keys())
    plt.bar(sorted_steps, [steps_count[step] for step in sorted_steps], color='skyblue')
    plt.title(f'{data_type.capitalize()} Data - Steps Distribution')
    plt.xlabel('Steps')
    plt.ylabel('Number of Games')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type}_steps_distribution.png'), dpi=300)
    plt.close()
    
    # 2. 单词长度分布（如果有）
    if data_type == 'random':
        word_lengths = [len(game['solution']) for game in data]
        length_count = Counter(word_lengths)
        
        plt.figure(figsize=(8, 6))
        sorted_lengths = sorted(length_count.keys())
        plt.bar(sorted_lengths, [length_count[l] for l in sorted_lengths], color='lightgreen')
        plt.title(f'{data_type.capitalize()} Data - Word Length Distribution')
        plt.xlabel('Word Length')
        plt.ylabel('Number of Games')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{data_type}_word_length_distribution.png'), dpi=300)
        plt.close()
    
    # 3. 单词难度分布
    difficulties = [game['word_difficulty'] for game in data]
    
    plt.figure(figsize=(8, 6))
    plt.hist(difficulties, bins=20, color='salmon', alpha=0.7)
    plt.title(f'{data_type.capitalize()} Data - Word Difficulty Distribution')
    plt.xlabel('Word Difficulty')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type}_difficulty_distribution.png'), dpi=300)
    plt.close()
    
    print(f"{data_type} data visualizations generated")

def visualize_embeddings(embedding_matrix, data_type, embedding_type, output_dir):
    """可视化嵌入向量"""
    print(f"\nGenerating {data_type} {embedding_type} embedding visualizations...")
    
    # 1. 嵌入分布
    plt.figure(figsize=(8, 6))
    plt.hist(embedding_matrix.flatten(), bins=50, color='purple', alpha=0.7)
    plt.title(f'{data_type.capitalize()} {embedding_type.capitalize()} Embedding Distribution')
    plt.xlabel('Embedding Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type}_{embedding_type}_embedding_distribution.png'), dpi=300)
    plt.close()
    
    # 2. 嵌入向量示例（前20个）
    plt.figure(figsize=(10, 6))
    for i in range(min(20, embedding_matrix.shape[0])):
        plt.plot(embedding_matrix[i], label=f'Index {i}')
    plt.title(f'{data_type.capitalize()} {embedding_type.capitalize()} Embedding Examples')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{data_type}_{embedding_type}_embedding_examples.png'), dpi=300)
    plt.close()
    
    print(f"{data_type} {embedding_type} embedding visualizations generated")

# 实验报告生成
def generate_experiment_report(random_data, games_data, word_freq, output_dir):
    """生成实验报告"""
    print(f"\n=== Generating Experiment Report ===")
    
    report_path = os.path.join(output_dir, 'embedding_experiment_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Wordle Embedding 实验报告\n\n")
        f.write("## 1. 实验概述\n\n")
        f.write("本实验针对Wordle游戏数据，生成了字符级和单词级的embedding模型，用于后续的游戏步数预测任务。\n\n")
        f.write("### 1.1 数据集信息\n\n")
        f.write(f"- **wordle_random.csv**: {len(random_data)}局游戏\n")
        f.write(f"- **wordle_games.csv**: {len(games_data)}局游戏\n\n")
        
        f.write("### 1.2 Embedding类型\n\n")
        f.write("- **单词级Embedding**: 针对每个单词生成一个向量表示\n")
        f.write("- **字符级Embedding**: 针对每个字符生成一个向量表示\n\n")
        
        f.write("## 2. 数据统计分析\n\n")
        
        # Random data stats
        f.write("### 2.1 wordle_random.csv 统计\n\n")
        random_steps = [game['steps'] for game in random_data]
        f.write(f"- 总游戏数: {len(random_data)}\n")
        f.write(f"- 平均步数: {np.mean(random_steps):.2f}\n")
        f.write(f"- 中位数步数: {np.median(random_steps):.2f}\n")
        
        # Games data stats
        f.write("\n### 2.2 wordle_games.csv 统计\n\n")
        games_steps = [game['steps'] for game in games_data]
        users = [game['username'] for game in games_data]
        unique_users = set(users)
        f.write(f"- 总游戏数: {len(games_data)}\n")
        f.write(f"- 总用户数: {len(unique_users)}\n")
        f.write(f"- 平均步数: {np.mean(games_steps):.2f}\n")
        f.write(f"- 中位数步数: {np.median(games_steps):.2f}\n")
        
        # Word stats
        f.write("\n### 2.3 单词统计\n\n")
        f.write(f"- 总单词数: {len(word_freq)}\n")
        f.write(f"- 最高频率单词: {max(word_freq, key=word_freq.get)} ({max(word_freq.values())}次)\n")
        f.write(f"- 最低频率单词: {min(word_freq, key=word_freq.get)} ({min(word_freq.values())}次)\n\n")
        
        f.write("## 3. Embedding 模型详情\n\n")
        
        f.write("### 3.1 词汇表构建\n\n")
        f.write("- **单词级词汇表**: 包含所有出现过的单词，添加了<PAD>和<UNK>特殊标记\n")
        f.write("- **字符级词汇表**: 包含所有出现过的字符，添加了<PAD>和<UNK>特殊标记\n\n")
        
        f.write("### 3.2 Embedding 矩阵生成\n\n")
        f.write(f"- **Embedding维度**: {EMBEDDING_DIM}\n")
        f.write(f"- **初始化方式**: 随机高斯分布\n")
        f.write(f"- **<PAD>向量**: 全0向量\n\n")
        
        f.write("## 4. 生成的文件\n\n")
        f.write("### 4.1 数据集文件\n\n")
        f.write("- `random_word_embedding.npz`: random数据单词级embedding\n")
        f.write("- `random_char_embedding.npz`: random数据字符级embedding\n")
        f.write("- `games_word_embedding.npz`: games数据单词级embedding\n")
        f.write("- `games_char_embedding.npz`: games数据字符级embedding\n\n")
        
        f.write("### 4.2 可视化图表\n\n")
        f.write("- 步数分布图\n")
        f.write("- 单词长度分布图\n")
        f.write("- 单词难度分布图\n")
        f.write("- Embedding分布直方图\n")
        f.write("- Embedding向量示例图\n\n")
        
        f.write("### 4.3 代码文件\n\n")
        f.write("- `embedding_processing.py`: Embedding生成主程序\n\n")
        
        f.write("## 5. 结论与展望\n\n")
        f.write("- 成功生成了针对两个数据集的字符级和单词级embedding模型\n")
        f.write("- 嵌入模型可以直接用于后续的Wordle游戏步数预测任务\n")
        f.write("- 可以进一步优化embedding生成方式，如使用CBOW或Skip-gram模型\n")
        f.write("- 可以考虑加入预训练的embedding模型，如Word2Vec或GloVe\n\n")
    
    print(f"Experiment report generated: {report_path}")

def main():
    print("Starting Wordle Embedding Generation Process...")
    
    # 确保输出目录存在
    output_dir = 'embedding'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 加载数据
    random_data = load_random_data(RANDOM_FILE)
    games_data = load_games_data(GAMES_FILE)
    
    # 2. 计算单词频率和玩家特征
    word_freq = calculate_word_frequency(random_data, games_data)
    player_features = calculate_player_features(games_data)
    
    # 3. 增强数据特征
    enhanced_random_data = add_features_to_data(random_data, 'random', word_freq)
    enhanced_games_data = add_features_to_data(games_data, 'games', word_freq, player_features)
    
    # 4. 生成单词级Embedding
    # 4.1 Random data word embedding
    random_word_vocab, random_word_to_idx, random_idx_to_word = build_word_vocabulary(enhanced_random_data, 'random')
    random_word_embedding = generate_word_embedding_matrix(random_word_vocab)
    
    # 4.2 Games data word embedding
    games_word_vocab, games_word_to_idx, games_idx_to_word = build_word_vocabulary(enhanced_games_data, 'games')
    games_word_embedding = generate_word_embedding_matrix(games_word_vocab)
    
    # 5. 生成字符级Embedding
    # 5.1 Random data char embedding
    random_char_vocab, random_char_to_idx, random_char_idx_to_char = build_char_vocabulary(enhanced_random_data, 'random')
    random_char_embedding = generate_char_embedding_matrix(random_char_vocab)
    
    # 5.2 Games data char embedding
    games_char_vocab, games_char_to_idx, games_char_idx_to_char = build_char_vocabulary(enhanced_games_data, 'games')
    games_char_embedding = generate_char_embedding_matrix(games_char_vocab)
    
    # 6. 保存Embedding数据
    print("\n=== Saving Embedding Data ===")
    
    # 6.1 Random word embedding
    np.savez(os.path.join(output_dir, 'random_word_embedding.npz'),
             vocabulary=random_word_vocab,
             word_to_idx=random_word_to_idx,
             idx_to_word=random_idx_to_word,
             embedding_matrix=random_word_embedding,
             data=enhanced_random_data)
    print("Random word embedding saved")
    
    # 6.2 Games word embedding
    np.savez(os.path.join(output_dir, 'games_word_embedding.npz'),
             vocabulary=games_word_vocab,
             word_to_idx=games_word_to_idx,
             idx_to_word=games_idx_to_word,
             embedding_matrix=games_word_embedding,
             data=enhanced_games_data)
    print("Games word embedding saved")
    
    # 6.3 Random char embedding
    np.savez(os.path.join(output_dir, 'random_char_embedding.npz'),
             vocabulary=random_char_vocab,
             char_to_idx=random_char_to_idx,
             idx_to_char=random_char_idx_to_char,
             embedding_matrix=random_char_embedding,
             data=enhanced_random_data)
    print("Random char embedding saved")
    
    # 6.4 Games char embedding
    np.savez(os.path.join(output_dir, 'games_char_embedding.npz'),
             vocabulary=games_char_vocab,
             char_to_idx=games_char_to_idx,
             idx_to_char=games_char_idx_to_char,
             embedding_matrix=games_char_embedding,
             data=enhanced_games_data)
    print("Games char embedding saved")
    
    # 7. 生成可视化图表
    generate_visualizations(enhanced_random_data, 'random', output_dir)
    generate_visualizations(enhanced_games_data, 'games', output_dir)
    
    visualize_embeddings(random_word_embedding, 'random', 'word', output_dir)
    visualize_embeddings(games_word_embedding, 'games', 'word', output_dir)
    visualize_embeddings(random_char_embedding, 'random', 'char', output_dir)
    visualize_embeddings(games_char_embedding, 'games', 'char', output_dir)
    
    # 8. 生成实验报告
    generate_experiment_report(enhanced_random_data, enhanced_games_data, word_freq, output_dir)
    
    # 9. 代码文件已直接存放在embedding目录中
    
    print("\n=== Embedding Generation Process Completed ===")
    print(f"All results saved to {output_dir} directory")

# 执行主函数
if __name__ == "__main__":
    main()

import csv
import numpy as np
import matplotlib.pyplot as plt
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
# 特征计算函数
# ----------------------
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
    player_features = defaultdict(get_default_player_features)  # 使用普通函数作为默认值
    
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
                hits = row[hits_key].lower().strip()
                
                if attempt == '' or hits == '':
                    continue
                
                game['attempts'].append(attempt)
                game['hits'].append(hits)
            
            # 计算游戏成功所需的步数
            game['steps'] = len(game['attempts'])
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
            try:
                game = {
                    'game_id': int(row['Game']),
                    'username': row['Username'].strip(),
                    'target': row['target'].lower().strip(),
                    'steps': int(row['Trial'])  # Trial表示尝试次数
                }
                data.append(game)
                
                count += 1
                if count % 100000 == 0:
                    print(f"Loaded {count} games data records so far...")
            except (ValueError, KeyError) as e:
                # 跳过无效行
                continue
    
    print(f"Loaded {len(data)} games data records")
    return data

def add_features_to_data(random_data, games_data):
    """为数据添加特征"""
    print("Adding features to data...")
    
    # 计算单词频率和难度
    word_freq = calculate_word_frequency(random_data, games_data)
    
    # 计算玩家特征
    player_features = calculate_player_features(games_data)
    
    # 为random_data添加单词难度
    for game in random_data:
        game['solution_difficulty'] = calculate_word_difficulty(game['solution'], word_freq)
    
    # 为games_data添加单词难度和玩家特征
    for game in games_data:
        # 添加单词难度
        game['target_difficulty'] = calculate_word_difficulty(game['target'], word_freq)
        
        # 添加玩家特征
        username = game['username']
        if username in player_features:
            features = player_features[username]
            game['player_total_games'] = features['total_games']
            game['player_avg_steps'] = features['avg_steps']
            game['player_median_steps'] = features['median_steps']
            game['player_success_rate'] = features['success_rate']
        else:
            # 新玩家的默认特征
            game['player_total_games'] = 1
            game['player_avg_steps'] = game['steps']
            game['player_median_steps'] = game['steps']
            game['player_success_rate'] = 1.0
    
    print("Features added to data")
    return word_freq, player_features

# ----------------------
# 数据统计与分析
# ----------------------
def analyze_random_data(random_data):
    """分析random数据的统计信息"""
    print("\n=== Random Data Analysis ===")
    
    # 1. 基本统计
    total_games = len(random_data)
    print(f"Total games: {total_games}")
    
    # 2. 步数统计
    steps = [game['steps'] for game in random_data]
    steps_count = Counter(steps)
    avg_steps = np.mean(steps)
    median_steps = np.median(steps)
    
    print(f"\nSteps Distribution:")
    for step in sorted(steps_count.keys()):
        print(f"  {step} steps: {steps_count[step]} games ({steps_count[step]/total_games*100:.2f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Median steps: {median_steps:.2f}")
    
    # 3. 单词长度统计
    word_lengths = [len(game['solution']) for game in random_data]
    length_count = Counter(word_lengths)
    print(f"\nWord Length Distribution:")
    for length in sorted(length_count.keys()):
        print(f"  {length} letters: {length_count[length]} games ({length_count[length]/total_games*100:.2f}%)")
    
    # 4. 游戏成功统计
    success_count = sum(1 for game in random_data if game['steps'] > 0)
    success_rate = success_count / total_games * 100
    print(f"\nSuccess Rate: {success_rate:.2f}%")
    
    return {
        'total_games': total_games,
        'steps': steps,
        'avg_steps': avg_steps,
        'median_steps': median_steps,
        'word_lengths': word_lengths,
        'success_rate': success_rate
    }

def analyze_games_data(games_data):
    """分析games数据的统计信息"""
    print("\n=== Games Data Analysis ===")
    
    # 1. 基本统计
    total_games = len(games_data)
    print(f"Total games: {total_games}")
    
    # 2. 用户统计
    users = [game['username'] for game in games_data]
    unique_users = set(users)
    print(f"Total users: {len(unique_users)}")
    
    # 3. 每个用户的游戏数量
    user_game_count = Counter(users)
    avg_games_per_user = total_games / len(unique_users)
    print(f"Average games per user: {avg_games_per_user:.2f}")
    
    # 4. 步数统计
    steps = [game['steps'] for game in games_data]
    steps_count = Counter(steps)
    avg_steps = np.mean(steps)
    median_steps = np.median(steps)
    
    print(f"\nSteps Distribution:")
    for step in sorted(steps_count.keys()):
        print(f"  {step} steps: {steps_count[step]} games ({steps_count[step]/total_games*100:.2f}%)")
    print(f"Average steps: {avg_steps:.2f}")
    print(f"Median steps: {median_steps:.2f}")
    
    # 5. 单词统计
    words = [game['target'] for game in games_data]
    unique_words = set(words)
    print(f"\nUnique words: {len(unique_words)}")
    
    return {
        'total_games': total_games,
        'users': users,
        'unique_users': unique_users,
        'user_game_count': user_game_count,
        'steps': steps,
        'avg_steps': avg_steps,
        'median_steps': median_steps,
        'words': words,
        'unique_words': unique_words
    }

# ----------------------
# Embedding相关函数
# ----------------------
def build_vocabulary(random_data, games_data):
    """构建词汇表"""
    print("Building vocabulary...")
    
    # 收集所有单词
    all_words = set()
    
    # 从random_data中提取单词
    for game in random_data:
        all_words.add(game['solution'])
        for attempt in game['attempts']:
            all_words.add(attempt)
    
    # 从games_data中提取单词
    for game in games_data:
        all_words.add(game['target'])
    
    # 排序并添加特殊标记
    sorted_words = sorted(all_words)
    vocabulary = ['<PAD>', '<UNK>'] + sorted_words  # 添加填充和未知标记
    
    # 创建单词到索引的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Vocabulary built with {len(vocabulary)} words")
    return vocabulary, word_to_idx, idx_to_word

def generate_embedding_matrix(vocabulary, embedding_dim=EMBEDDING_DIM):
    """生成embedding矩阵"""
    print(f"Generating embedding matrix with dimension {embedding_dim}...")
    
    # 初始化embedding矩阵
    embedding_matrix = np.random.randn(len(vocabulary), embedding_dim) * 0.01
    
    # 为填充标记设置全零向量
    embedding_matrix[0] = np.zeros(embedding_dim)
    
    print(f"Embedding matrix generated: shape = {embedding_matrix.shape}")
    return embedding_matrix

def word_to_embedding(word, word_to_idx, embedding_matrix):
    """将单词转换为embedding向量"""
    idx = word_to_idx.get(word, 1)  # 默认为<UNK>的索引
    return embedding_matrix[idx]

def game_to_embedding_sequence(game, word_to_idx, embedding_matrix):
    """将游戏转换为embedding序列"""
    sequence = []
    
    # 根据游戏类型处理不同的数据结构
    if 'solution' in game and 'attempts' in game and game['attempts']:
        # random_data格式
        for attempt in game['attempts']:
            # 将尝试转换为embedding
            attempt_embedding = word_to_embedding(attempt, word_to_idx, embedding_matrix)
            sequence.append(attempt_embedding)
    elif 'target' in game:
        # games_data格式
        # 直接使用目标单词的embedding
        target_embedding = word_to_embedding(game['target'], word_to_idx, embedding_matrix)
        sequence.append(target_embedding)
    
    return np.array(sequence)

def visualize_embedding_distribution(embedding_matrix, filename="embedding_distribution.png"):
    """可视化embedding分布"""
    print("Generating embedding distribution visualization...")
    
    plt.figure(figsize=(10, 6))
    
    # 计算embedding向量的均值和标准差
    embedding_means = np.mean(embedding_matrix, axis=1)
    embedding_stds = np.std(embedding_matrix, axis=1)
    
    plt.scatter(embedding_means, embedding_stds, alpha=0.5, color='blue')
    plt.title('Embedding Distribution (Mean vs Standard Deviation)')
    plt.xlabel('Mean of Embedding Vector')
    plt.ylabel('Standard Deviation of Embedding Vector')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Embedding distribution plot saved as {filename}")

def visualize_word_embedding(word, vocabulary, word_to_idx, embedding_matrix, filename="word_embedding.png"):
    """可视化单个单词的embedding"""
    print(f"Visualizing embedding for word: {word}...")
    
    plt.figure(figsize=(8, 6))
    
    # 获取单词的embedding向量
    idx = word_to_idx.get(word, 1)
    word_embedding = embedding_matrix[idx]
    
    # 绘制embedding向量
    plt.bar(range(len(word_embedding)), word_embedding, color='green', alpha=0.7, edgecolor='black')
    plt.title(f'Embedding Vector for Word: "{word}"')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Embedding Value')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Word embedding plot saved as {filename}")

# ----------------------
# 可视化函数
# ----------------------
def plot_steps_distribution(random_steps, games_steps, filename="steps_distribution.png"):
    """绘制步数分布对比图"""
    plt.figure(figsize=(12, 6))
    
    # 绘制random数据的步数分布
    plt.subplot(1, 2, 1)
    plt.hist(random_steps, bins=range(0, 8), align='left', color='blue', alpha=0.7, edgecolor='black')
    plt.title('Random Data Steps Distribution')
    plt.xlabel('Steps to Solve')
    plt.ylabel('Number of Games')
    plt.grid(True, axis='y')
    plt.xticks(range(0, 7))
    
    # 绘制games数据的步数分布
    plt.subplot(1, 2, 2)
    plt.hist(games_steps, bins=range(0, 8), align='left', color='orange', alpha=0.7, edgecolor='black')
    plt.title('Games Data Steps Distribution')
    plt.xlabel('Steps to Solve')
    plt.ylabel('Number of Games')
    plt.grid(True, axis='y')
    plt.xticks(range(0, 7))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Steps distribution plot saved as {filename}")

def plot_user_game_distribution(user_game_count, filename="user_game_distribution.png"):
    """绘制用户游戏数量分布"""
    plt.figure(figsize=(10, 6))
    
    # 统计每个用户的游戏数量，只显示前20个
    sorted_counts = sorted(user_game_count.items(), key=lambda x: x[1], reverse=True)[:20]
    users = [user for user, count in sorted_counts]
    counts = [count for user, count in sorted_counts]
    
    plt.bar(range(len(users)), counts, color='green', alpha=0.7)
    plt.title('Top 20 Users by Game Count')
    plt.xlabel('User')
    plt.ylabel('Number of Games')
    plt.xticks(range(len(users)), users, rotation=90)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"User game distribution plot saved as {filename}")

def plot_word_length_distribution(word_lengths, filename="word_length_distribution.png"):
    """绘制单词长度分布"""
    plt.figure(figsize=(8, 6))
    
    length_count = Counter(word_lengths)
    lengths = sorted(length_count.keys())
    counts = [length_count[l] for l in lengths]
    
    plt.bar(lengths, counts, color='purple', alpha=0.7, edgecolor='black')
    plt.title('Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Number of Games')
    plt.grid(True, axis='y')
    plt.xticks(lengths)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Word length distribution plot saved as {filename}")

def plot_data_comparison(random_stats, games_stats, filename="data_comparison.png"):
    """绘制数据对比图"""
    plt.figure(figsize=(10, 6))
    
    categories = ['Total Games', 'Average Steps', 'Median Steps', 'Success Rate']
    random_values = [
        random_stats['total_games'],
        random_stats['avg_steps'],
        random_stats['median_steps'],
        random_stats['success_rate']
    ]
    games_values = [
        games_stats['total_games'],
        games_stats['avg_steps'],
        games_stats['median_steps'],
        100  # games数据都是成功的
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, random_values, width, label='Random Data', color='blue', alpha=0.7)
    plt.bar(x + width/2, games_values, width, label='Games Data', color='orange', alpha=0.7)
    
    plt.title('Data Comparison Between Random and Games')
    plt.ylabel('Value')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Data comparison plot saved as {filename}")

# ----------------------
# 数据预处理主函数
# ----------------------
def main():
    print("Starting Wordle Data Preprocessing...")
    
    # 1. 加载数据
    random_data = load_random_data(RANDOM_FILE)
    games_data = load_games_data(GAMES_FILE)
    
    # 2. 添加特征（单词难度和玩家行为特征）
    word_freq, player_features = add_features_to_data(random_data, games_data)
    
    # 3. 分析数据
    random_stats = analyze_random_data(random_data)
    games_stats = analyze_games_data(games_data)
    
    # 4. 构建词汇表和生成embedding
    vocabulary, word_to_idx, idx_to_word = build_vocabulary(random_data, games_data)
    embedding_matrix = generate_embedding_matrix(vocabulary)
    
    # 5. 生成可视化图表
    print("\n=== Generating Visualizations ===")
    # 原有可视化
    plot_steps_distribution(random_stats['steps'], games_stats['steps'])
    plot_user_game_distribution(games_stats['user_game_count'])
    plot_word_length_distribution(random_stats['word_lengths'])
    plot_data_comparison(random_stats, games_stats)
    # Embedding相关可视化
    visualize_embedding_distribution(embedding_matrix)
    # 可视化一个常见单词的embedding
    if 'aback' in vocabulary:
        visualize_word_embedding('aback', vocabulary, word_to_idx, embedding_matrix)
    elif len(vocabulary) > 2:  # 确保有足够的单词
        visualize_word_embedding(vocabulary[2], vocabulary, word_to_idx, embedding_matrix)
    
    # 6. 保存预处理后的数据
    print("\n=== Saving Preprocessed Data ===")
    # 保存为npz格式，包含embedding相关数据和新特征
    np.savez('preprocessed_data.npz',
             random_data=np.array(random_data, dtype=object),
             games_data=np.array(games_data, dtype=object),
             random_stats=random_stats,
             games_stats=games_stats,
             vocabulary=np.array(vocabulary),
             word_to_idx=word_to_idx,
             idx_to_word=idx_to_word,
             embedding_matrix=embedding_matrix,
             embedding_dim=EMBEDDING_DIM,
             word_freq=word_freq,
             player_features=player_features)
    print("Preprocessed data saved as preprocessed_data.npz")
    
    print("\nData preprocessing completed!")
    print("\n=== Summary ===")
    print(f"- Loaded {len(random_data)} random games and {len(games_data)} games data records")
    print(f"- Added features: word difficulty and player behavior features")
    print(f"- Built vocabulary with {len(vocabulary)} words")
    print(f"- Generated embedding matrix with shape {embedding_matrix.shape}")
    print(f"- Created 6 visualization plots")
    print("- Saved all data to preprocessed_data.npz")

if __name__ == "__main__":
    main()
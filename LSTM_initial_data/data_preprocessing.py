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

# 文件路径
RANDOM_FILE = r'e:\资料\NJU Journey\课程类\正式课程\3-1 大数据系统原理与应用\期末大作业\wordle_random.csv\wordle_random.csv'
GAMES_FILE = r'e:\资料\NJU Journey\课程类\正式课程\3-1 大数据系统原理与应用\期末大作业\wordle_games.csv\wordle_games.csv'

# 数据加载函数
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
    
    # 统计每个Username出现的次数
    username_counts = Counter([game['username'] for game in data])
    print(f"Total unique usernames before filtering: {len(username_counts)}")
    
    # 仅保留Username出现次数大于等于5次的行
    filtered_data = [game for game in data if username_counts[game['username']] >= 5]
    print(f"Loaded {len(filtered_data)} games data records after filtering (Username count >= 5)")
    
    return filtered_data

# 数据统计与分析
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

# 可视化函数
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

# 数据预处理主函数
def main():
    print("Starting Wordle Data Preprocessing...")
    
    # 1. 加载数据
    random_data = load_random_data(RANDOM_FILE)
    games_data = load_games_data(GAMES_FILE)
    
    # 2. 分析数据
    random_stats = analyze_random_data(random_data)
    games_stats = analyze_games_data(games_data)
    
    # 3. 生成可视化图表
    print("\n=== Generating Visualizations ===")
    plot_steps_distribution(random_stats['steps'], games_stats['steps'])
    plot_user_game_distribution(games_stats['user_game_count'])
    plot_word_length_distribution(random_stats['word_lengths'])
    plot_data_comparison(random_stats, games_stats)
    
    # 4. 保存预处理后的数据
    print("\n=== Saving Preprocessed Data ===")
    # 保存为npz格式
    np.savez('preprocessed_data.npz',
             random_data=np.array(random_data, dtype=object),
             games_data=np.array(games_data, dtype=object),
             random_stats=random_stats,
             games_stats=games_stats)
    print("Preprocessed data saved as preprocessed_data.npz")
    
    print("\nData preprocessing completed!")

if __name__ == "__main__":
    main()
import requests

# 获取网页内容
url = 'http://localhost:8501'
r = requests.get(url)

# 保存到文件
output_path = 'dashboard/wordle_dashboard.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(r.text)

print(f"Dashboard saved to {output_path}")

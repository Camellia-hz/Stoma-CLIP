import json
import random
from collections import defaultdict

def split_jsonl_by_label(input_file, train_file, test_file, ratio=0.8, seed=42):
    """
    按标签分层划分数据集
    
    参数：
        input_file: 输入jsonl文件路径
        train_file: 训练集输出路径
        test_file: 测试集输出路径
        ratio: 训练集比例
        seed: 随机种子
    """
    # 初始化数据结构
    label_data = defaultdict(list)
    
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            label_data[item['label']].append(item)
    
    for k, v in label_data.items():
        print(f"{k}: {len(v)}")
    

    train_set = []
    test_set = []
    
    random.seed(seed)
    
    for label, items in label_data.items():
        random.shuffle(items)

        if len(items) < 10:
            train_set.extend(items)
            test_set.extend(items)
        else:
            split_idx = int(len(items) * ratio)
            train_set.extend(items[:split_idx])
            test_set.extend(items[split_idx:])
    
    with open(train_file, 'w') as f:
        for item in train_set:
            f.write(json.dumps(item) + '\n')
    
    with open(test_file, 'w') as f:
        for item in test_set:
            f.write(json.dumps(item) + '\n')

# 使用示例
if __name__ == "__main__":
    split_jsonl_by_label(
        input_file='data/single_symptoms_data.jsonl',
        train_file='single_symptoms_train.jsonl',
        test_file='single_symptoms_test.jsonl',
        ratio=0.8,
        seed=2023
    )
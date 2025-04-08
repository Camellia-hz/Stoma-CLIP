import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import json
import sys
sys.path.append('.')
import pmc_clip
from training.params import parse_args
from training.data import PmcDataset
from training.fusion_method import convert_model_to_cls 


# 标签映射
LABEL_MAP = {
    "Irritant dermatitis": 0,
    "Allergic contact dermatitis": 1,
    "Mechanical injury": 2,
    "Folliculitis": 3,
    "Fungal infection": 4,
    "Skin hyperplasia": 5,
    "Parastomal varices": 6,
    "Urate crystals": 7,
    "Cancerous metastasis": 8,
    "Pyoderma gangrenosum": 9,
    "Normal": 10
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def main():
    # 创建输出目录
    output_dir = './evaluation_results_pmc_clip_cat'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型配置
    model_path = "logs/0321-Stoma-clip-train-cls/2025_03_21-23_45_18-model_RN50_fusion4-lr_1e-05-b_256-j_8-p_amp/checkpoints/epoch_150.pt"
    model_name = "RN50_fusion4"
    args = parse_args()
    args.model = model_name
    args.pretrained = model_path
    args.device = device
    args.mlm = True
    args.train_data = "data/single_symptoms_test.jsonl"
    args.image_dir = "./data/cleaned_data"
    args.csv_img_key = "image"
    args.csv_caption_key = "caption"
    args.context_length = 77
    args.num_classes = len(LABEL_MAP)
    args.output_dir = output_dir
    
    # 创建模型和预处理函数
    model, _, preprocess = pmc_clip.create_model_and_transforms(args)
    model = convert_model_to_cls(model, num_classes=args.num_classes, fusion_method='concat')
    
    # 加载模型权重
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    
    state_dict_real = {}
    for k, v in state_dict['state_dict'].items():
        state_dict_real[k.replace("module.", "", 1)] = v
    print(model.load_state_dict(state_dict_real))
    model.to(device=device)
    
    # 准备数据集
    dataset = PmcDataset(args,
                         input_filename=args.train_data,
                         transforms=preprocess,
                         is_train=False)
    
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"测试集样本数: {len(dataset)}")
    
    # 收集预测结果
    all_preds = []
    all_probs = []
    all_labels = []
    
    print("开始评估...")
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            labels = batch["cls_label"].to(device)
            
            # 前向传播
            outputs = model(batch)
            
            # 获取预测结果
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算整体AUC（使用one-vs-rest策略的平均）
    try:
        y_true_bin = label_binarize(all_labels, classes=range(args.num_classes))
        if args.num_classes == 2:
            overall_fpr, overall_tpr, _ = roc_curve(y_true_bin[:, 1], all_probs[:, 1])
            overall_auc = roc_auc_score(y_true_bin, all_probs[:, 1])
        else:
            overall_fpr, overall_tpr, _ = roc_curve(y_true_bin.ravel(), all_probs.ravel())
            overall_auc = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='micro')
    except Exception as e:
        print(f"计算整体AUC时出错: {e}")
        return
    
    # 保存整体ROC曲线数据
    roc_data = {
        "fpr": overall_fpr.tolist(),
        "tpr": overall_tpr.tolist(),
        "auc": overall_auc
    }
    roc_file = os.path.join(output_dir, "overall_roc_data.json")
    with open(roc_file, "w") as f:
        json.dump(roc_data, f)
    print(f"整体ROC曲线数据已保存至: {roc_file}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(overall_fpr, overall_tpr, label=f"Overall (AUC = {overall_auc:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Overall ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"整体ROC曲线图已保存至: {os.path.join(output_dir, 'overall_roc_curve.png')}")


if __name__ == '__main__':
    main()
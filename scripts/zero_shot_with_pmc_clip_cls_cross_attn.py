import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
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
    output_dir = './evaluation_results_pmc_clip_cross_attn_no_finetune_with_stoma_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型配置
    model_path = "logs/0322-Stoma-clip-train-cls-cross-attention/2025_03_29-11_54_10-model_RN50_fusion4-lr_1e-05-b_256-j_8-p_amp/checkpoints/epoch_200.pt"
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
    model = convert_model_to_cls(model, num_classes=args.num_classes, fusion_method='cross_attention')
    
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
            # images = batch["images"].to(device)
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
    
    # 计算并显示混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12), dpi=300)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=[REVERSE_LABEL_MAP[i] for i in range(args.num_classes)],
                    yticklabels=[REVERSE_LABEL_MAP[i] for i in range(args.num_classes)])

    # 旋转标签以提高可读性
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=45, fontsize=10)

    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    # 计算多分类性能指标
    print("\n===== 性能指标 =====")
    
    # 计算每个类的指标
    class_metrics = []
    
    # 整体精度
    accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    print(f"整体准确率: {accuracy:.4f}")
    
    # 汇总所有类别的混淆矩阵元素
    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0
    
    # 针对每个类计算指标
    for class_idx in range(args.num_classes):
        # 将多分类问题转换为二分类问题（一对多）
        binary_labels = (all_labels == class_idx).astype(int)
        binary_preds = (all_preds == class_idx).astype(int)
        
        # 计算混淆矩阵元素
        TP = np.sum((binary_labels == 1) & (binary_preds == 1))
        FP = np.sum((binary_labels == 0) & (binary_preds == 1))
        TN = np.sum((binary_labels == 0) & (binary_preds == 0))
        FN = np.sum((binary_labels == 1) & (binary_preds == 0))
        
        # 累加到总量
        total_TP += TP
        total_FP += FP
        total_TN += TN
        total_FN += FN
        
        # 计算指标
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        ppv = TP / (TP + FP) if (TP + FP) > 0 else 0
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0
        plr = sensitivity / (1 - specificity) if (1 - specificity) > 0 else 0
        nlr = (1 - sensitivity) / specificity if specificity > 0 else 0
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        # 计算AUC
        fpr, tpr, _ = roc_curve(binary_labels, all_probs[:, class_idx])
        roc_auc = auc(fpr, tpr)
        
        class_metrics.append({
            'class': class_idx,
            'accuracy': (TP + TN) / (TP + TN + FP + FN),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'plr': plr,
            'nlr': nlr,
            'f1': f1,
            'auc': roc_auc
        })
    
    # 计算整体指标（使用累加的混淆矩阵元素）
    overall_metrics = {
        'accuracy': accuracy,  # 使用已计算的整体准确率
        'sensitivity': total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0,
        'specificity': total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0,
        'ppv': total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0,
        'npv': total_TN / (total_TN + total_FN) if (total_TN + total_FN) > 0 else 0,
        'plr': (total_TP / (total_TP + total_FN)) / (1 - (total_TN / (total_TN + total_FP))) if (total_TN + total_FP) > 0 and (total_TP + total_FN) > 0 else 0,
        'nlr': (1 - (total_TP / (total_TP + total_FN))) / (total_TN / (total_TN + total_FP)) if (total_TN + total_FP) > 0 and (total_TP + total_FN) > 0 else 0,
        'f1': 2 * ((total_TP / (total_TP + total_FP)) * (total_TP / (total_TP + total_FN))) / ((total_TP / (total_TP + total_FP)) + (total_TP / (total_TP + total_FN))) if (total_TP + total_FP) > 0 and (total_TP + total_FN) > 0 else 0
    }
    
    # 计算整体AUC（使用one-vs-rest策略的平均）
    try:
        y_true_bin = label_binarize(all_labels, classes=range(args.num_classes))
        if args.num_classes == 2:
            overall_metrics['auc'] = roc_auc_score(y_true_bin, all_probs[:, 1])
        else:
            overall_metrics['auc'] = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='micro')
    except Exception as e:
        overall_metrics['auc'] = np.mean([m['auc'] for m in class_metrics])  # 回退到平均AUC
        print(f"计算整体AUC时出错: {e}，使用平均值代替")
    
    # 打印每个类别的指标
    print("\n每个类别的性能指标:")
    for i, metrics in enumerate(class_metrics):
        print(f"\n类别 {i}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  敏感性: {metrics['sensitivity']:.4f}")
        print(f"  特异性: {metrics['specificity']:.4f}")
        print(f"  PPV(精确率): {metrics['ppv']:.4f}")
        print(f"  NPV: {metrics['npv']:.4f}")
        print(f"  PLR: {metrics['plr']:.4f}")
        print(f"  NLR: {metrics['nlr']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
    
    # 打印整体指标
    print("\n整体评估指标:")
    print(f"  准确率: {overall_metrics['accuracy']:.4f}")
    print(f"  敏感性: {overall_metrics['sensitivity']:.4f}")
    print(f"  特异性: {overall_metrics['specificity']:.4f}")
    print(f"  PPV(精确率): {overall_metrics['ppv']:.4f}")
    print(f"  NPV: {overall_metrics['npv']:.4f}")
    print(f"  PLR: {overall_metrics['plr']:.4f}")
    print(f"  NLR: {overall_metrics['nlr']:.4f}")
    print(f"  F1分数: {overall_metrics['f1']:.4f}")
    print(f"  AUC: {overall_metrics['auc']:.4f}")
    
    # 绘制ROC曲线
    plt.figure(figsize=(14, 12), dpi=300)

    # 为每个类别绘制ROC曲线
    for i, metrics in enumerate(class_metrics):
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        plt.plot(fpr, tpr, label=f'{REVERSE_LABEL_MAP[i]} (AUC = {metrics["auc"]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Multi-class ROC Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {os.path.join(args.output_dir, 'roc_curves.png')}")
    
    # 保存结果到文本文件
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"模型: {args.model}\n")
        f.write(f"权重: {args.pretrained}\n")
        f.write(f"测试集: {args.train_data}\n\n")
        
        f.write("===== 性能指标 =====\n")
        f.write(f"整体准确率: {accuracy:.4f}\n\n")
        
        f.write("每个类别的性能指标:\n")
        for i, metrics in enumerate(class_metrics):
            f.write(f"\n类别 {i}:\n")
            f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"  敏感性: {metrics['sensitivity']:.4f}\n")
            f.write(f"  特异性: {metrics['specificity']:.4f}\n")
            f.write(f"  PPV(精确率): {metrics['ppv']:.4f}\n")
            f.write(f"  NPV: {metrics['npv']:.4f}\n")
            f.write(f"  PLR: {metrics['plr']:.4f}\n")
            f.write(f"  NLR: {metrics['nlr']:.4f}\n")
            f.write(f"  F1分数: {metrics['f1']:.4f}\n")
            f.write(f"  AUC: {metrics['auc']:.4f}\n")
        
        # 整体指标
        f.write("\n整体评估指标:\n")
        f.write(f"  准确率: {overall_metrics['accuracy']:.4f}\n")
        f.write(f"  敏感性: {overall_metrics['sensitivity']:.4f}\n")
        f.write(f"  特异性: {overall_metrics['specificity']:.4f}\n")
        f.write(f"  PPV(精确率): {overall_metrics['ppv']:.4f}\n")
        f.write(f"  NPV: {overall_metrics['npv']:.4f}\n")
        f.write(f"  PLR: {overall_metrics['plr']:.4f}\n")
        f.write(f"  NLR: {overall_metrics['nlr']:.4f}\n")
        f.write(f"  F1分数: {overall_metrics['f1']:.4f}\n")
        f.write(f"  AUC: {overall_metrics['auc']:.4f}\n")
    
    print(f"\n评估结果已保存至 {os.path.join(args.output_dir, 'evaluation_results.txt')}")

if __name__ == '__main__':
    main()
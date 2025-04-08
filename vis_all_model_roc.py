import os
import json
import matplotlib.pyplot as plt

def load_roc_data(roc_files):
    """
    加载多个模型的ROC数据
    参数:
        roc_files (list): 包含多个模型的overall_roc_data.json文件路径的列表
    返回:
        roc_data_list (list): 包含每个模型的ROC数据字典的列表
    """
    roc_data_list = []
    for roc_file in roc_files:
        with open(roc_file, "r") as f:
            roc_data = json.load(f)
            roc_data_list.append(roc_data)
    return roc_data_list

def plot_combined_roc(roc_data_list, model_names, output_path):
    """
    绘制多个模型的ROC曲线到同一张图中
    参数:
        roc_data_list (list): 包含每个模型的ROC数据字典的列表
        model_names (list): 每个模型的名称列表
        output_path (str): 保存ROC曲线图的路径
    """
    plt.figure(figsize=(10, 8))
    
    for roc_data, model_name in zip(roc_data_list, model_names):
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc_value = roc_data["auc"]
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.4f})")
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    
    # 图形设置
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Combined ROC Curves for Multiple Models', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC曲线图已保存至: {output_path}")

def main():
    # 定义存放多个模型ROC数据的目录
    roc_data_dir = "./roc_result"  # 替换为实际路径
    output_path = "./combined_roc_curve.png"  # 保存最终ROC曲线图的路径
    
    # 获取所有模型的overall_roc_data.json文件路径
    roc_files = [os.path.join(roc_data_dir, f) for f in os.listdir(roc_data_dir)]
    
    # 模型名称（从文件名提取）
    model_names = [os.path.basename(f).replace(".json", "") for f in roc_files]
    
    # 加载所有模型的ROC数据
    roc_data_list = load_roc_data(roc_files)
    
    # 绘制并保存组合ROC曲线
    plot_combined_roc(roc_data_list, model_names, output_path)

if __name__ == "__main__":
    main()
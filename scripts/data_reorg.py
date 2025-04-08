import os
import shutil
import uuid
import json
from pathlib import Path

def process_multi_annotations(input_root, output_dir):
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历所有标注批次（第一次标注、第二次标注等）
    for phase in Path(input_root).iterdir():
        if not phase.is_dir(): continue
        
        # 递归处理每个标注批次中的文件夹
        for json_path in phase.rglob("*.json"):
            # 获取配对图片文件（支持多格式）
            img_candidates = list(json_path.parent.glob(f"{json_path.stem}.*"))
            img_candidates = [f for f in img_candidates if f.suffix.lower() in ['.jpg','.png','.jpeg']]
            
            if len(img_candidates) != 1:
                print(f"⚠️ 文件匹配异常: {json_path} 找到 {len(img_candidates)} 个图片")
                continue
            
            img_path = img_candidates[0]
            
            # 生成UUID新文件名
            new_name = uuid.uuid4().hex  # 生成32字符短格式UUID
            new_img_name = f"{new_name}{img_path.suffix}"
            new_json_name = f"{new_name}.json"
            
            # 复制并重命名图片
            shutil.copy2(img_path, Path(output_dir)/new_img_name)
            
            # 修改并保存JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["imagePath"] = new_img_name
            data.pop('imageData')
            
            with open(Path(output_dir)/new_json_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    process_multi_annotations(
        input_root = "data",          # 原始数据根目录
        output_dir = "cleaned_data"   # 统一输出目录
    )
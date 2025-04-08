import json
import os
from pathlib import Path

LABEL_MAPPING = {
    # 简单病症
    "刺激性皮炎": {
        "label_en": "Irritant dermatitis",
        "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas"
    },
    "过敏性接触性皮炎": {
        "label_en": "Allergic contact dermatitis",
        "symptoms": "Edema, blisters, exudate, well-defined erythema, pruritus"
    },
    "机械性损伤": {
        "label_en": "Mechanical injury",
        "symptoms": "Epidermal breakdown, irregular erythema, bleeding, exudate, pain"
    },
    "毛囊炎": {
        "label_en": "Folliculitis",
        "symptoms": "Follicle-centered erythema, pustules, papules, dry crusts, pruritus, pain"
    },
    "真菌感染": {
        "label_en": "Fungal infection",
        "symptoms": "Well-circumscribed erythema, pustules, gray-white scaling, pus, pruritus"
    },
    "皮肤增生": {
        "label_en": "Skin hyperplasia",
        "symptoms": "Thickening/verrucous projections, hyperpigmentation, pruritus, pain"
    },
    "静脉曲张": {
        "label_en": "Parastomal varices",
        "symptoms": "Dusky blue skin, radiating veins, bleeding"
    },
    "尿酸盐结晶": {
        "label_en": "Urate crystals",
        "symptoms": "Brown/gray crystalline deposits, strong urine odor, bleeding, hematuria, pruritus"
    },
    "癌性转移": {
        "label_en": "Cancerous metastasis",
        "symptoms": "Abnormal mass, foul odor, bleeding, pain"
    },
    "坏疽性脓皮病": {
        "label_en": "Pyoderma gangrenosum",
        "symptoms": "Pustules, deep ulcers, irregular purple edges, exudate, pain"
    },
    
    # # 复合病症
    # "刺激性皮炎并发机械性损伤": {
    #     "label_en": "Irritant dermatitis with mechanical injury",
    #     "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas, bleeding"
    # },
    # "刺激性皮炎并发尿酸盐结晶": {
    #     "label_en": "Irritant dermatitis with urate crystals",
    #     "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas, brown/gray deposits, strong urine odor, bleeding, hematuria, pruritus"
    # },
    # "刺激性皮炎并发皮肤增生": {
    #     "label_en": "Irritant dermatitis with skin hyperplasia",
    #     "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas, thickening/verrucous projections, hyperpigmentation, pruritus"
    # },
    # "刺激性皮炎合并坏疽性脓皮病": {
    #     "label_en": "Irritant dermatitis with pyoderma gangrenosum",
    #     "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas, pustules, deep ulcers, irregular purple edges"
    # },
    # "刺激性皮炎损伤并发毛囊炎": {
    #     "label_en": "Irritant dermatitis with folliculitis",
    #     "symptoms": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas, follicle-centered erythema, pustules, papules, dry crusts, pruritus"
    # },
    # "机械性损伤并发毛囊炎": {
    #     "label_en": "Mechanical injury with folliculitis",
    #     "symptoms": "Follicle-centered erythema, pustules, papules, dry crusts, pruritus, pain, epidermal breakdown, irregular erythema, bleeding, exudate"
    # },
    "正常": {
        "label_en": "Normal",
        "symptoms": "Normal skin appearance"
    }
}

def generate_master_json(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 遍历处理每个JSON文件
        for json_path in Path(input_dir).glob("*.json"):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取有效标签
            for label in [k for k, v in data['flags'].items() if v]:
                # 构建条目并写入
                try:
                    entry = {
                        "image": data['imagePath'],
                        "caption": LABEL_MAPPING[label]["symptoms"],
                        "label": LABEL_MAPPING[label]["label_en"]
                    }
                    # 写入单行JSON
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                except:
                    continue

if __name__ == "__main__":
    generate_master_json(
        input_dir="data/cleaned_data",
        output_file="single_symptoms_data.jsonl"
    )
    
    
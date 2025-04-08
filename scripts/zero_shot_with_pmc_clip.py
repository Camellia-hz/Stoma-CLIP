import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import sys
import math
sys.path.append('.')
import pmc_clip
from training.params import parse_args
from training.data import PmcDataset

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ori ckpt: PMC-CLIP/ckpt/pmc_clip.pt
model_path = "logs/0320-Stoma-clip-train/2025_03_20-21_49_25-model_RN50_fusion4-lr_1e-06-b_128-j_8-p_amp/checkpoints/epoch_500.pt"
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
args.context_length= 77


model, _, preprocess = pmc_clip.create_model_and_transforms(args)
model.to(device)
tokenizer = model.tokenizer


MEDICAL_DESCRIPTIONS = {
    "Irritant dermatitis": "Exudate, epidermal breakdown, irregular erythema, pain, confined to contact areas",
    "Allergic contact dermatitis": "Edema, blisters, exudate, well-defined erythema, pruritus",
    "Mechanical injury": "Epidermal breakdown, irregular erythema, bleeding, exudate, pain",
    "Folliculitis": "Follicle-centered erythema, pustules, papules, dry crusts, pruritus, pain",
    "Fungal infection": "Well-circumscribed erythema, pustules, gray-white scaling, pus, pruritus",
    "Skin hyperplasia": "Thickening/verrucous projections, hyperpigmentation, pruritus, pain",
    "Parastomal varices": "Dusky blue skin, radiating veins, bleeding",
    "Urate crystals": "Brown/gray crystalline deposits, strong urine odor, bleeding, hematuria, pruritus",
    "Cancerous metastasis": "Abnormal mass, foul odor, bleeding, pain",
    "Pyoderma gangrenosum": "Pustules, deep ulcers, irregular purple edges, exudate, pain",
    "Normal": "Normal skin appearance"
}

text_prompts = [
    f"{MEDICAL_DESCRIPTIONS[label]}." 
    for label in LABEL_MAP.keys()
]

dataset = PmcDataset(args,
                     input_filename=args.train_data,
                     transforms=preprocess,
                     is_train=False)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)


encoded_input = tokenizer(text_prompts, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
input_ids = encoded_input['input_ids'].to(device)
x = model.text_encoder(
    input_ids=input_ids,
    output_attentions = False
).last_hidden_state
last_token_index = torch.nonzero((input_ids == model.cls_id).squeeze())
text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]
text_features = text_features @ model.text_projection
text_features = text_features / text_features.norm(dim=-1, keepdim=True)
logit_scale = 4.4292


accuracy_dict = {
    'correct': 0,
    'total': 0,
    'class_correct': [0] * len(LABEL_MAP),
    'class_total': [0] * len(LABEL_MAP)
}

model.eval()

with torch.no_grad(), torch.cuda.amp.autocast():
    for batch in tqdm(data_loader):
        images = batch['images'].to(device)
        labels = batch['cls_label'].to(device)

        image_features = model.encode_image(images)['image_features']
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * math.exp(logit_scale)
        _, predicted = torch.max(logits, dim=1)

        correct = (predicted == labels)
        accuracy_dict['correct'] += correct.sum().item()
        accuracy_dict['total'] += labels.size(0)

        for cls_idx in range(len(LABEL_MAP)):
            mask = (labels == cls_idx)
            accuracy_dict['class_correct'][cls_idx] += correct[mask].sum().item()
            accuracy_dict['class_total'][cls_idx] += mask.sum().item()

total = accuracy_dict['total']
acc = accuracy_dict['correct'] / total
print(f"\nOverall Accuracy: {acc*100:.2f}%")

print("\nClass-wise Accuracy:")
for cls_name, cls_idx in LABEL_MAP.items():
    correct = accuracy_dict['class_correct'][cls_idx]
    total = accuracy_dict['class_total'][cls_idx]
    acc = correct / total if total > 0 else 0
    print(f"{cls_name:<25}: {acc*100:.2f}% ({correct}/{total})")
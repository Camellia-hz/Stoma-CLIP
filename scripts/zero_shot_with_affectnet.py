import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('.')
import pmc_clip
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# /data/oss_bucket_0/Users/haozhou/checkpoints/openai/ViT-L-14-336px.pt
# /data/oss_bucket_0/Users/haozhou/runs/open_clip/logs_faceptor/2025_02_02-10_36_07-model_ViT-L-14-336-lr_6e-06-b_128-j_4-p_amp/checkpoints/epoch_32_faceptor.pt
model_path = "/mnt/workspace/haozhou/runs/open_clip/logs_more_datasets_repeat_8rafdb/2025_02_14-01_44_45-model_ViT-L-14-336-lr_6e-06-b_128-j_4-p_amp/checkpoints/epoch_32.pt"
model_name = "ViT-L-14-336"

model, _, preprocess = pmc_clip.create_model_and_transforms(model_name=model_name, pretrained=model_path)

tokenizer = pmc_clip.get_tokenizer(model_name)
model.to(device)

dataset = AffectNetDataset(data_path='/data/oss_bucket_0/Users/haozhou/data/FaceDataset/AffectNet',
                       transforms=preprocess,
                       is_train=False,
                       tokenizer=tokenizer,
                       return_labels=True)

data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Only keep the 'Emotion' part for zero-shot ability testing
texts_dict = {
    'Emotion': [
        "A person with neutral expression.",            # 0
        "A person smiling happily.",                    # 1
        "A person looking sad.",                        # 2
        "A person with surprised expression.",          # 3
        "A person showing fear.",                       # 4
        "A person with disgusted expression.",          # 5
        "An angry-looking person.",                     # 6
        "A person showing contempt."                    # 7
    ]
}

# Tokenize the text dictionary for Emotion
text_tokens_dict = {
    key: tokenizer(texts).to(device) 
    for key, texts in texts_dict.items()
}

# Initialize accuracy dictionary for Emotion
accuracy_dict = {
    'Emotion': {
        'correct': 0,
        'total': 0,
        'class_correct': [0] * len(texts_dict['Emotion']),
        'class_total': [0] * len(texts_dict['Emotion'])
    }
}

model.eval()

with torch.no_grad(), torch.cuda.amp.autocast():
    for batch in tqdm(data_loader):
        images, _, labels = batch
        images = images.to(device)
        
        # Only process Emotion key
        key = 'Emotion'
        text_tokens = text_tokens_dict[key]
        raw_label = labels.to(device)

        label = raw_label.long()

        # Get image and text features
        image_features = model.encode_image(images)
        text_features = model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100

        _, predicted = torch.max(logits, dim=1)

        correct = (predicted == label)
        accuracy_dict[key]['correct'] += correct.sum().item()
        accuracy_dict[key]['total'] += label.size(0)
                
        for cls_idx in range(len(texts_dict['Emotion'])):
            mask = (label == cls_idx)
            accuracy_dict[key]['class_correct'][cls_idx] += correct[mask].sum().item()
            accuracy_dict[key]['class_total'][cls_idx] += mask.sum().item()

# Calculate and print Emotion accuracy
emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

total = accuracy_dict['Emotion']['total']
acc = accuracy_dict['Emotion']['correct'] / total
print(f"\nEmotion整体准确率: {acc*100:.2f}%")

print("各类别准确率:")
for cls_idx, cls_name in enumerate(emotion_labels):
    cls_correct = accuracy_dict['Emotion']['class_correct'][cls_idx]
    cls_total = accuracy_dict['Emotion']['class_total'][cls_idx]  # Use the accumulated total
    cls_acc = cls_correct / cls_total if cls_total > 0 else 0
    print(f"{cls_name}: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")
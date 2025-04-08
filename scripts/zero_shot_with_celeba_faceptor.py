import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from PIL import Image

sys.path.append('.')
import open_clip
from open_clip_train.data import CelebADatasetWithFaceptor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "/mnt/nfs3/haozhou/codes/open_clip/pretrained/ViT-L-14-336px.pt"
model_name = "ViT-L-14-336"
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)


dataset = CelebADatasetWithFaceptor(
    input_path='/mnt/nfs3/haozhou/codes/open_clip/data/CelebA',
    transforms=preprocess,
    tokenizer=tokenizer,
    split='test',
    return_labels=True,
    emotion_file='/mnt/nfs3/haozhou/codes/open_clip/data/CelebA/Anno/predictions_attr_with_faceptor.txt'
)
data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

texts_dict = {
    'Male': ["This person is male.", "This person is female."],
    'Emotion': [
        "A person with neutral expression.",            # 0
        "A person smiling happily.",                    # 1
        "A person looking sad.",                        # 2
        "A person with surprised expression.",          # 3
        "A person showing fear.",                       # 4
        "A person with disgusted expression.",          # 5
        "An angry-looking person.",                     # 6
        "A person showing contempt."                    # 7
    ],
    'Eyeglasses': ["This person is wearing eyeglasses.", "This person is not wearing eyeglasses."],
    'Young': ["This person is young.", "This person is not young."],
    'Heavy_Makeup': ["This person is wearing heavy makeup.", "This person is not wearing heavy makeup."],
    'High_Cheekbones': ["This person has high cheekbones.", "This person does not have high cheekbones."]
}

text_tokens_dict = {
    key: tokenizer(texts).to(device) 
    for key, texts in texts_dict.items()
}

accuracy_dict = {
    key: {
        'correct': 0, 
        'total': 0,
        'class_correct': [0]*len(texts) if key == 'Emotion' else None,
        'class_total': [0]*len(texts) if key == 'Emotion' else None
    } 
    for key, texts in texts_dict.items()
}
model.eval()
# import pdb; pdb.set_trace()
with torch.no_grad(), torch.cuda.amp.autocast():
    for batch in tqdm(data_loader):
        images, _, labels = batch
        images = images.to(device)
        
        for key in texts_dict:
            text_tokens = text_tokens_dict[key]
            raw_label = labels[key].to(device)
            
            if key != 'Emotion':
                label = (raw_label == -1).long()
            else:
                label = raw_label.long()

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ text_features.T) * 100
            
            if key == 'Emotion':
                _, predicted = torch.max(logits, dim=1)
            else:
                probs = logits.softmax(dim=-1)
                _, predicted = torch.max(probs, dim=1)

            correct = (predicted == label)
            accuracy_dict[key]['correct'] += correct.sum().item()
            accuracy_dict[key]['total'] += label.size(0)
                
            if key == 'Emotion':
                for cls_idx in range(len(texts_dict['Emotion'])):
                    mask = (label == cls_idx)
                    accuracy_dict[key]['class_correct'][cls_idx] += correct[mask].sum().item()
                    accuracy_dict[key]['class_total'][cls_idx] += mask.sum().item() 

emotion_labels = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]

for key in accuracy_dict:
    if key == 'Emotion':
        total = accuracy_dict[key]['total']
        acc = accuracy_dict[key]['correct'] / total
        print(f"\nEmotion整体准确率: {acc*100:.2f}%")
        
        print("各类别准确率:")
        for cls_idx, cls_name in enumerate(emotion_labels):
            cls_correct = accuracy_dict[key]['class_correct'][cls_idx]
            cls_total = accuracy_dict[key]['class_total'][cls_idx]  # 使用累计总数
            cls_acc = cls_correct / cls_total if cls_total > 0 else 0
            print(f"{cls_name}: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")
    else:
        acc = accuracy_dict[key]['correct'] / accuracy_dict[key]['total']
        print(f"{key}准确率: {acc*100:.2f}%")
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('.')
import open_clip
from open_clip_train.data import CelebADataset
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/mnt/nfs3/haozhou/codes/open_clip/pretrained/epoch_24.pt"
model_name = "ViT-L-14-336"

model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=model_path)
tokenizer = open_clip.get_tokenizer(model_name)
model.to(device)

dataset = CelebADataset(input_path='/mnt/nfs3/haozhou/codes/open_clip/data/CelebA', 
                        transforms=preprocess,
                        tokenizer=tokenizer,
                        split='test',
                        return_labels=True,
                        emotion_file='/mnt/nfs3/haozhou/codes/open_clip/data/CelebA/Anno/predictions_attr_with_faceptor.txt')

data_loader = DataLoader(dataset, batch_size=128, shuffle=False)

texts_dict = {
    'Male': ["This person is male.", "This person is female."],
    'Smiling': ["This person is smiling.", "This person is not smiling."],
    'Eyeglasses': ["This person is wearing eyeglasses.", "This person is not wearing eyeglasses."],
    'Young': ["This person is young.", "This person is not young."],
    'Heavy_Makeup': ["This person is wearing heavy makeup.", "This person is not wearing heavy makeup."],
    'High_Cheekbones': ["This person has high cheekbones.", "This person does not have high cheekbones."]
}

text_tokens_dict = {key: tokenizer(texts).to(device) for key, texts in texts_dict.items()}

accuracy_dict = {key: {'correct': 0, 'total': 0} for key in texts_dict}

model.eval()

with torch.no_grad(), torch.cuda.amp.autocast():
    for images, image_descriptions, labels in tqdm(data_loader):
        images = images.to(device)
        
        for key, text_tokens in text_tokens_dict.items():
            label = labels[key].to(device)
            label = (label == -1).long()  # 将1变为0，-1变为1

            image_features = model.encode_image(images)
            text_features = model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, predicted = torch.max(text_probs, dim=-1)

            accuracy_dict[key]['correct'] += (predicted == label).sum().item()
            accuracy_dict[key]['total'] += label.size(0)

for key in accuracy_dict:
    accuracy = accuracy_dict[key]['correct'] / accuracy_dict[key]['total']
    print(f"Accuracy for {key}: {accuracy * 100:.2f}%")
from sre_parse import State
import torch
from typing import Optional, Tuple
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
from PIL import Image
import matplotlib.pyplot as plt
import math

import sys
sys.path.append('.')
import pmc_clip
from training.params import parse_args


state = torch.load("data/pmc_oa/text_encoder.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/mnt/csi-data-aly/user/haozhou/Projects/research/PMC-CLIP/ckpt/pmc_clip.pt"
model_name = "RN50_fusion4"
args = parse_args()
args.model = model_name
args.pretrained = model_path
args.device = device
args.mlm = True

model, _, preprocess = pmc_clip.create_model_and_transforms(args)
model.to(device)
tokenizer = model.tokenizer

def _convert_to_rgb(image):
    return image.convert('RGB')

def image_transform(
        image_size: int,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        fill_color: int = 0,
):
    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    mean = mean or (0.48145466, 0.4578275, 0.40821073)  # OpenAI dataset mean
    std = std or (0.26862954, 0.26130258, 0.27577711)  # OpenAI dataset std
    normalize = Normalize(mean=mean, std=std)

    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
    ]
    transforms.extend([
        _convert_to_rgb,
        ToTensor(),
        normalize,
    ])
    return Compose(transforms)


# Load Image
preprocess_val = image_transform(
    image_size=224,
)

image_path_ls = [
    '/mnt/csi-data-aly/user/haozhou/Projects/research/PMC-CLIP/data/pmc_oa/chest_X-ray.jpg',
    '/mnt/csi-data-aly/user/haozhou/Projects/research/PMC-CLIP/data/pmc_oa/brain_MRI.jpg'
]
images = []
image_tensor = []
for image_path in image_path_ls:
    image = Image.open(image_path).convert('RGB')
    images.append(image)
    image_tensor.append(preprocess_val(image))

image_tensor = torch.stack(image_tensor, dim=0).to(device)


# Extract Image feature
image_feature = model.encode_image(image_tensor)
if isinstance(image_feature, dict):
    image_feature = image_feature['image_features']

print(f'\033[32mimage size\033[0m: {image_tensor.shape}; feature size: {image_feature.shape}')

bert_input = [
    'chest X-ray',
    'brain MRI',
]
encoded_input = tokenizer(bert_input, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
input_ids = encoded_input['input_ids'].to(device)

# Extract Text feature
# text_feature = model.text_encoder(input_ids)
# last_hidden_state = text_feature.last_hidden_state
# pooler_output = text_feature.pooler_output
# text_feature = pooler_output @ model.text_projection
x = model.text_encoder(
    input_ids=input_ids,
    output_attentions = False
).last_hidden_state
last_token_index = torch.nonzero((input_ids == model.cls_id).squeeze())
text_features = x[torch.arange(x.shape[0]), last_token_index[:, 1]]
text_feature = text_features @ model.text_projection
print("\033[32mtext_feature.shape\033[0m ", text_feature.shape)


# Logit Scale
logit_scale = 4.4292
image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
similarity = (math.exp(logit_scale) * image_feature @ text_feature.T).softmax(dim=-1)

for i, (image_path, image) in enumerate(zip(image_path_ls, images)):
    print(image_path)
    plt.imshow(image)
    plt.show()
    for j in range(len(bert_input)):
        print(f'{bert_input[j]}: {similarity[i, j].item()}')
    print('\n')
    

"""
chest_X-ray.jpg
chest X-ray: 0.9877390265464783
brain MRI: 0.01226091105490923

brain_MRI.jpg
chest X-ray: 0.16185268759727478
brain MRI: 0.8381473422050476
"""

"""
chest_X-ray.jpg
chest X-ray: 0.9999970197677612
brain MRI: 2.979377995870891e-06

brain_MRI.jpg
chest X-ray: 7.547180658207253e-10
brain MRI: 1.0

"""
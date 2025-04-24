import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, image_features, text_features):
        # 形状调整为 [seq_len, batch, dim]
        image_features = image_features.unsqueeze(0)
        text_features = text_features.unsqueeze(0)
        
        # 交叉注意力: 图像作为查询，文本作为键和值
        attn_output, _ = self.attention(
            query=self.norm1(image_features),
            key=self.norm2(text_features),
            value=text_features
        )
        
        # 合并特征
        fused = attn_output.squeeze(0) + image_features.squeeze(0)
        return fused

class BiCrossAttentionFusion(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.img2text_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.text2img_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, image_features, text_features):
        # 准备形状
        img_feats = image_features.unsqueeze(0)
        txt_feats = text_features.unsqueeze(0)
        
        # 图像→文本注意力
        img2txt, _ = self.img2text_attn(
            query=self.norm1(img_feats),
            key=self.norm2(txt_feats),
            value=txt_feats
        )
        
        # 文本→图像注意力
        txt2img, _ = self.text2img_attn(
            query=self.norm2(txt_feats),
            key=self.norm1(img_feats),
            value=img_feats
        )
        
        # 合并结果
        img2txt = img2txt.squeeze(0)
        txt2img = txt2img.squeeze(0)
        
        fused = self.fusion_mlp(torch.cat([img2txt, txt2img], dim=1))
        return fused


class FiLMFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.film_generator = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.ReLU(),
            nn.Linear(dim*2, dim*2)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, image_features, text_features):
        # 文本生成调制参数
        film_params = self.film_generator(text_features)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # 调制图像特征
        modulated = self.norm(image_features) * (1 + gamma) + beta
        
        # 最终融合
        fused = modulated + text_features
        return fused


class GatedFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(dim*2, dim),
            nn.Sigmoid()
        )
        self.proj_img = nn.Linear(dim, dim)
        self.proj_txt = nn.Linear(dim, dim)
        
    def forward(self, image_features, text_features):
        # 投影特征
        img_proj = self.proj_img(image_features)
        txt_proj = self.proj_txt(text_features)
        
        # 计算门控权重
        gate = self.gate_net(torch.cat([image_features, text_features], dim=1))
        
        # 门控融合
        fused = gate * img_proj + (1 - gate) * txt_proj
        return fused

def convert_model_to_cls(model, num_classes=11, fusion_method='concat'):
    in_features = 768
    
    # 添加融合模块
    if fusion_method == 'cross_attention':
        model.fusion = CrossAttentionFusion(dim=768)
    elif fusion_method == 'bi_cross_attention':
        model.fusion = BiCrossAttentionFusion(dim=768)
    elif fusion_method == 'film':
        model.fusion = FiLMFusion(dim=768)
    elif fusion_method == 'gated':
        model.fusion = GatedFusion(dim=768)
    elif fusion_method == 'concat':
        in_features = 768 * 2
        model.fusion = lambda img, txt: torch.cat([img, txt], dim=1)
    
    # 修改分类器输入维度
    model.cls_mlp = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    if hasattr(model.fusion, 'to'):
        model.fusion.to(device=model.device)
    model.cls_mlp.to(device=model.device)
    
    # 保存原始forward
    original_forward = model.forward
    
    def new_forward(self, batch):
        clip_prediction = original_forward(batch)
        
        # 使用选择的融合方法
        fused_features = self.fusion(
            clip_prediction["image_features"],
            clip_prediction["text_features"]
        )
        
        cls_output = self.cls_mlp(fused_features)
        return cls_output
    
    model.forward = new_forward.__get__(model, model.__class__)
    
    # 冻结原模型参数
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    # 解冻融合模块和分类器参数
    if hasattr(model.fusion, 'parameters'):
        for param in model.fusion.parameters():
            param.requires_grad = True
    for param in model.cls_mlp.parameters():
        param.requires_grad = True
    
    # 检查模型
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"Trainable parameter: {name}")
    
    return model



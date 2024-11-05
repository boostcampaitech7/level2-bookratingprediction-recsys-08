import torch
import torch.nn as nn
from .VisionTransformer import VisionTransformer  # VisionTransformer import
from .TextBERT import TextBERT  # TextBERT import

class MultiModalModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=10, embed_dim=768):
        super().__init__()
        # Vision Transformer (이미지 처리)
        self.vit = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=embed_dim)
        # TextBERT (텍스트 처리)
        self.bert = TextBERT(embed_dim=embed_dim)
        # 이미지 임베딩과 텍스트 임베딩을 결합한 후 분류를 위한 fully connected layer
        self.fc = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, image, input_ids, attention_mask):
        # 이미지 임베딩 계산 (Vision Transformer 사용)
        image_emb = self.vit(image)
        # 텍스트 임베딩 계산 (BERT 사용)
        text_emb = self.bert(input_ids, attention_mask)
        # 이미지 임베딩과 텍스트 임베딩을 결합
        combined_emb = torch.cat((image_emb, text_emb), dim=1)
        # 결합된 임베딩을 통해 최종 분류
        output = self.fc(combined_emb)
        return output
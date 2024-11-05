import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import nn

# PatchEmbedding: 이미지를 패치 단위로 나누어 임베딩 벡터로 변환하는 클래스
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, num_patches, embed_dim)
        return x

# TransformerEncoder: 트랜스포머 인코더, 여러 레이어로 구성, 각 레이어가 트랜스포머 인코더 레이어로 입력 벡터 처리
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, hidden_dim=3072, num_layers=12, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# VisionTransformer: 전체 모델. 패치 임베딩, 클래스 토큰 추가, 포지셔널 임베딩 이후 트랜스포머 인코더로 처리
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, num_heads=12, hidden_dim=3072, num_layers=12, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(embed_dim, num_heads, hidden_dim, num_layers, dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embedding(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        x = self.encoder(x)
        x = x[:, 0]  # cls_token 추출
        x = self.mlp_head(x)
        return x

# 테스트
model = VisionTransformer(img_size=224, patch_size=16, num_classes=10)
input_tensor = torch.randn(8, 3, 224, 224)  # 8개의 224x224 RGB 이미지 배치
output = model(input_tensor)
print(output.shape)  # (8, 10)
import torch
import torch.nn as nn
from transformers import BertModel

class TextBERT(nn.Module):
    def __init__(self, bert_model_name="bert-base-uncased", embed_dim=768):
        super(TextBERT, self).__init__()
        # Pretrained BERT 모델을 로드하고 그 출력을 ViT의 embed_dim에 맞추기 위한 fully connected layer 추가
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        # BERT 모델에 input_ids와 attention_mask를 전달하여 출력 얻음
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [CLS] 토큰의 출력 임베딩만 사용 (outputs.last_hidden_state[:, 0, :]는 [CLS] 토큰에 해당)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # 임베딩 크기를 VisionTransformer의 embed_dim에 맞추기 위해 변환
        cls_output = self.fc(cls_output)
        return cls_output
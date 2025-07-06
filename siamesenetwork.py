import torch 
import torch.nn as nn
from transformers import AutoModel

class SiameseNetwork(nn.Module):
    def __init__(self, model_name="roberta-base", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        rep1 = self.encoder(input_ids=input_ids_1, attention_mask=attention_mask_1).last_hidden_state[:, 0, :]
        rep2 = self.encoder(input_ids=input_ids_2, attention_mask=attention_mask_2).last_hidden_state[:, 0, :]

        combined = torch.cat([rep1, rep2], dim=1)
        out = self.classifier(self.dropout(combined))
        return out.squeeze(1)
from typing import Optional

import torch
import torch.nn as nn
from transformers import DistilBertModel


class DistilBertClassifier(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(
            "distilbert-base-multilingual-cased"
        )
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(self.distilbert.config.hidden_size, output_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distilbert_outputs = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = distilbert_outputs[0][:, 0, :]
        output = self.fc(self.dropout(pooled))

        return output

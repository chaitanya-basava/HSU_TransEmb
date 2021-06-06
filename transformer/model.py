import torch
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, model_name, hidden_states=256, dropout=0.5, n_classes=2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.lin0 = nn.Linear(self.bert.config.hidden_size, hidden_states)
        self.lin1 = nn.Linear(hidden_states, n_classes)
        self.reLU = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # output = output.last_hidden_state[:, 0, :]
        output = output.last_hidden_state
        output = torch.mean(output, dim=1)
        y = self.lin0(output)
        y = self.reLU(y)
        y = self.dropout(y)
        return self.lin1(y)

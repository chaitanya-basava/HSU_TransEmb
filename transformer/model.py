import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, model_name, hidden_states=256, n_classes=2):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.lin0 = nn.Linear(self.bert.config.hidden_size, hidden_states)
        self.lin1 = nn.Linear(hidden_states, n_classes)
        self.lReLU = nn.LeakyReLU(0.1)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        y = self.lin0(bert_output["pooler_output"])
        y = self.lReLU(y)
        return self.lin1(y)

import torch
import torch.nn as nn
from transformers import BertModel
import pdb


class BertDownstream(nn.Module):
    def __init__(self, output_dim=1, avg_hs_output=True,
                 transformer_type='bert-base-uncased', hidden_dim=768,
                 local=False, device='cpu'):
        super().__init__()
        self.local = local
        self.device = device
        self.output_dim = output_dim
        self.avg_hs_output = avg_hs_output
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained(transformer_type)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input_ids, attention_mask):
        '''
        input:
            input_ids: BATCH X SEQ_LEN
            attention_mask: BATCH X SEQ_LEN
        return:
            logits: Logits tensor of shape BATCH X NUM_LABELS
        '''
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        pooler_output = outputs[1]
        if self.avg_hs_output:
            avg_hs_out = torch.mean(last_hidden_state, dim=1)
            logits = self.fc(avg_hs_out)
        else:
            logits = self.fc(pooler_output)
        return logits



import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torch
from dgl.nn import SAGEConv, GATConv


# RoBerta with linear layer
class RobertaMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_ratio, freeze=False):
        super(RobertaMultiLabelClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_ratio)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.out(pooled_output)
        return logits, pooled_output

class BertMultiLabelClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_ratio, output='logits'):
        super(BertMultiLabelClassification, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_ratio)
        self.output = output
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        if self.output == 'logits':
            pooled_output = self.dropout(pooled_output)
            logits = self.out(pooled_output)
            return logits, pooled_output
        else:
            return outputs[0], pooled_output

# RoBerta with Label Attention
# ref) "A Label Attention Model for ICD Coding from Clinical Text." Vu, Thanh, Dat Quoc Nguyen, and Anthony Nguyen.(IJCAI, 2020)
class RobertaLabelAttention(nn.Module):
    def __init__(self, model_name, num_labels, dropout_ratio, freeze=False):
        super(RobertaLabelAttention, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        # Layers for fine-tuning:
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256, bias=True)
        self.fc2 = nn.Linear(256, 28, bias=True)

        self.dropout = nn.Dropout(dropout_ratio)
        # Module_list
        self.linears = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, 1)
            for _ in range(num_labels)
        ])      
        # bert parameter freeze
        if freeze == True:
            print("Bert Freeze")
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, data, mask, **kwargs):
        outputs_data = self.bert(input_ids=data, attention_mask=mask, **kwargs)
        pooled_output = outputs_data['pooler_output'] # temp 
        token_output = outputs_data[0] # [batch, seqlen, hidden]
        # Label Attention
        z = F.tanh(self.fc1(token_output))
        att_weight = torch.softmax(self.fc2(z), dim=1).transpose(1, 2) # [batch, num_label, seqlen]
        label_output = torch.bmm(att_weight, token_output)# [batch, num_label, hidden]
        # Module_list
        logits = [self.linears[i](label_output[:, i, :]) for i in range(self.num_labels)] # [batch, 1] * 35
        logits_cat = torch.cat([logits[i] for i in range(self.num_labels)], dim=1)
        return logits_cat
        

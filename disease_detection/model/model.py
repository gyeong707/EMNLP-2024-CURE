import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel, AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.util import read_json
from model.sngp import mean_field_logits


#-----------------------------------Baseline-----------------------------------#
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


class Symp(nn.Module):
    def __init__(self, input_dim, filter_num=50, filter_sizes=[1], dropout=0.2, num_labels=5, max_pooling_k=1):
        super(Symp, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.max_pooling_k = max_pooling_k
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, filter_num, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_labels) # multi

    def forward(self, input_seqs, seq_masks=None):
        # input_seqs: (batch_size, seq_len, input_dim)
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [self.kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits, x

    def kmax_pooling(self, x, k):
        return x.sort(dim = 2)[0][:, :, -k:]


class SympWithFactor(nn.Module):
    def __init__(self, input_dim, filter_num=50, filter_sizes=[1], dropout=0.2, num_labels=5, max_pooling_k=1):
        super(SympWithFactor, self).__init__()
        self.filter_num = filter_num
        self.filter_sizes = filter_sizes
        self.hidden_size = len(filter_sizes) * filter_num
        self.max_pooling_k = max_pooling_k
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, filter_num, size) for size in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size, num_labels) # multi

    def forward(self, symptom, factor, seq_masks=None):
        # input_seqs: (batch_size, seq_len, input_dim)
        input_seqs = torch.cat([symptom, factor], dim=1)
        x = [F.relu(conv(input_seqs)) for conv in self.convs]
        x = [self.kmax_pooling(item, self.max_pooling_k).mean(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits, x

    def kmax_pooling(self, x, k):
        return x.sort(dim = 2)[0][:, :, -k:]



#------------------------------Ours-----------------------------# 
class CURE(nn.Module):
    def __init__(self, target_models, target_uncertainty, predictors, hidden_dim, num_labels, num_models, dropout_ratio, include_gpt, freeze=True):
        super(CURE, self).__init__()
        # Hyper-parameters
        self.include_gpt = include_gpt
        self.num_labels = num_labels
        # Optional: Include reesult of GPT-4o
        if self.include_gpt:
            self.num_models = num_models + 1
        else: 
            self.num_models = num_models
        self.num_uncertainty = len(target_uncertainty)
        self.hidden_dim = hidden_dim
        self.mean_field_factor = 0.1 # default
        # Model predictions:: sub-models
        self.target_models_name = target_models
        self.target_uncertainty = target_uncertainty
        self.predictors = predictors
        # Uncertainty-aware decision fusion layers
        self.linear1 = nn.Linear((self.num_models * self.num_labels)+self.num_uncertainty, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.logits = nn.Linear(self.hidden_dim, self.num_labels)
        self.dropout = nn.Dropout(dropout_ratio)
        # Optional: Freeze sub-models
        self.reset_parameters()
        if freeze:
            for model in self.predictors:
                for name, param in model.named_parameters():
                    param.requires_grad = False

    def reset_parameters(self): # Initialize weights
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.logits.weight)

    def make_prediction(self, model_type, predictor, input_data):
        if model_type == "bertq":
            input_ids, attention_mask = input_data 
            logits, cov = predictor(input_ids, attention_mask, return_gp_cov=True, update_cov=False)
            logits = mean_field_logits(logits, cov, mean_field_factor=self.mean_field_factor)
            logits = torch.sigmoid(logits)
            return logits

        if model_type == "bertc":
            input_ids, attention_mask = input_data
            logits, cov = predictor(input_ids, attention_mask, return_gp_cov=True, update_cov=False)
            logits = mean_field_logits(logits, cov, mean_field_factor=self.mean_field_factor)
            logits = torch.sigmoid(logits)
            return logits

        elif model_type == "symp":
            logits, _ = predictor(input_data)
            return torch.sigmoid(logits)

        elif model_type == "sympc":
            swfactor, _ = input_data
            logits, _ = predictor(swfactor)
            return torch.sigmoid(logits)

    def forward(self, text_input_ids, text_attention_mask, 
                factor_input_ids, factor_attention_mask, 
                symptom, factors, symptom_uncertainty, gpt_pred):
        # Setting variables
        swfactor = torch.cat([symptom, factors], dim=1).unsqueeze(2)
        symptom = symptom.unsqueeze(2)
        uncertainties = []
        pred_logits = []

        # Model predictions
        with torch.no_grad():
            for model_type, predictor in zip(self.target_models_name, self.predictors):
                # Set input data
                if model_type == "bertq": input_data = (text_input_ids, text_attention_mask)
                elif model_type == "bertc": input_data = (factor_input_ids, factor_attention_mask)
                elif model_type == "symp": input_data = symptom
                elif model_type == "sympc": input_data = (swfactor, factors)
                else: raise ValueError("Invalid Model Type")
                # Return logits
                if model_type == "bertq" or model_type == "bertc":
                    pred_logit = self.make_prediction(model_type, predictor, input_data)
                    pred_logits.append(pred_logit)
                elif model_type == "symp":
                    pred_logit = self.make_prediction(model_type, predictor, input_data)
                    pred_logits.append(pred_logit)
                else:
                    pred_logit = self.make_prediction(model_type, predictor, input_data)
                    pred_logits.append(pred_logit)

        # Uncertainty-aware decision fusion
        uncertainties.append(symptom_uncertainty.unsqueeze(1))
        uncertainties = torch.cat(uncertainties, dim=1) # [batch size, num_uncertainty]

        if self.include_gpt:
            pred_logits.append(gpt_pred)

        # stacked = torch.stack(pred_logits, dim=2) # [batch_size, num_labels, num_models]
        stacked_all = torch.cat(pred_logits, dim=1) # [batch_size, num_labels * num_models]
        cat = torch.cat([uncertainties, stacked_all], dim=1) # [batch_size, (num_labels * num_models) + num_uncertainty]
        
        if self.target_uncertainty == []:
            hidden = self.linear1(stacked_all)
            hidden = F.elu(self.dropout(hidden))
            hidden = self.linear2(hidden)
            hidden = F.elu(self.dropout(hidden))
            logits = self.logits(hidden)
        else:   
            hidden = self.linear1(cat)
            hidden = F.elu(self.dropout(hidden))
            hidden = self.linear2(hidden)
            hidden = F.elu(self.dropout(hidden))
            logits = self.logits(hidden)

        # print("[Predictions]: \n", stacked[0], stacked.size())
        # print("[Logits]: ", F.sigmoid(logits[0]), logits.size())
        return logits
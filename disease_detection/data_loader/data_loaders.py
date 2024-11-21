import ast
import torch
from torch.utils.data import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import model.model as module_arch
from model.sngp import SNGP

# BERT-based model
class BertDataset(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = self.X[idx]
        label = torch.tensor(self.y[idx], dtype=torch.float32)
        
        self.encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        item = {
            'input_ids': self.encoding['input_ids'].flatten(),
            'attention_mask': self.encoding['attention_mask'].flatten(),
            'labels': label
        }
        return item


class SymptomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X # symptom feature: one-hot
        self.y = y # disease label: one-hot

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        symptom = self.X[idx].unsqueeze(1)
        label = torch.FloatTensor(self.y[idx])
        item = {
            'symptom': symptom,
            'labels': label
        }
        return item
    
class SymptomFactorDataset(Dataset):
    def __init__(self, symptom, factor, y):
        self.symptom = symptom
        self.factor = factor
        self.y = y

    def __len__(self):
        return len(self.symptom)
    
    def __getitem__(self, idx):
        symptom = self.symptom[idx].unsqueeze(1)
        factor = self.factor[idx].unsqueeze(1) 
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        item = {
            'symptom': symptom,
            'factor': factor,
            'labels': label
        }
        return item

class OursDataset(Dataset):
    def __init__(self, tokenizer, max_length, text, symptom, factor, factor_text, uncertainty, gpt_pred, y):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text = text
        self.factor_text = factor_text
        self.symptom = symptom
        self.factor = factor
        self.gpt_pred = gpt_pred
        self.uncertainty = uncertainty
        self.y = y

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        symptom = self.symptom[idx]  
        factor = self.factor[idx]  
        factor_text = self.factor_text[idx]
        gpt_pred = self.gpt_pred[idx]
        uncertainty = torch.tensor(self.uncertainty[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.float32)

        self.encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        self.factor_encoding = self.tokenizer.encode_plus(
            factor_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        item = {
            'text_input_ids': self.encoding['input_ids'].flatten(),
            'text_attention_mask': self.encoding['attention_mask'].flatten(),
            'factor_input_ids': self.factor_encoding['input_ids'].flatten(),
            'factor_attention_mask': self.factor_encoding['attention_mask'].flatten(),
            'symptom': symptom,
            'factor': factor,
            'gpt_pred': gpt_pred,
            'uncertainty': uncertainty,
            'labels': label
        }
        return item



#-----------------------------------External_Functions------------------------------------#
def multilabel_stratified_split(data, label, seed, fold, n_splits=5, shuffle=True, mode='default', column_name='all_factors'):
    y = label
    if mode == 'symp' or mode == 'phq9': X = data # symptom vector
    else: X = data[column_name].values  # sentence
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, test_index) in enumerate(mskf.split(X, y)):
        print(i, fold)
        if fold == i:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            return X_train, X_test, y_train, y_test, train_index, test_index

def create_label_index(data, label_cols, num_labels):
    data[label_cols] = data[label_cols].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x) # 0518 최고운 수정
    # data[label_cols] = data[label_cols].map(lambda x: ast.literal_eval(x))
    labels = torch.zeros(len(data), num_labels)
    for i in range(len(data)):
        target = data[label_cols][i]
        if target == []: pass
        else:
            for t in target:
                labels[i][t] = 1
    return labels

def revert_label_index(labels, threshold):
    labels = (labels > threshold) 
    data = []
    for i in range(labels.shape[0]):
        row = labels[i]
        indices = (row == 1).nonzero().flatten().tolist()
        data.append(indices)
    return data


#-----------------------------------Symptom:Functions------------------------------------#
def return_label_list(data, label_cols):
    # This function takes in a dataset and the names of label columns.
    # It extracts the labels from the specified columns and returns a sorted list of unique label.
    labels_list = []
    data[label_cols] = data[label_cols].map(lambda x: ast.literal_eval(x))
    for i in range(len(data)):
        for value in data[label_cols][i]:
            labels_list.append(value)
    label_list = list(set(labels_list))
    label_list.sort()
    return label_list

def make_dictionary(label_list):
    # This function takes in a list of labels and creates two dictionaries: LABEL_TO_LETTER and LETTER_TO_LABEL.
    LABEL_TO_LETTER = {i:v for i, v in enumerate(label_list)}
    LETTER_TO_LABEL = {v:k for k, v in LABEL_TO_LETTER.items()}
    return LABEL_TO_LETTER, LETTER_TO_LABEL

def create_symptom_index(data, label_cols, num_labels, s_dict):
    # This function takes in a dataset, the names of label columns, the number of labels, and a symptom dictionary.
    # It creates a label index tensor for the data, where each row represents a data instance and each column represents a label.
    # The values in the tensor indicate the presence (1) or absence (0) of each label for each instance.
    labels = torch.zeros(len(data), num_labels)
    for i in range(len(data)):
        target = data[label_cols][i]
        if target == []: pass
        else:
            for t in target:
                labels[i][s_dict[t]] = 1
    return labels


#-----------------------------------Ours:Functions------------------------------------#
def load_models(model_type, num_labels, fold, device):
    if model_type == 'bertq':
        model_dict = torch.load("./saved/models/BERT/fold_"+str(fold)+"/model_best.pth")
        backbone = module_arch.BertMultiLabelClassification("klue/bert-base", num_labels, 0.2, "pooler")
        encoder = SNGP(backbone, num_classes=num_labels, device=device)
    if model_type == 'bertc':
        model_dict = torch.load("./saved/models/BERT_with_context/fold_"+str(fold)+"/model_best.pth")
        backbone = module_arch.BertMultiLabelClassification("klue/bert-base", num_labels, 0.2, "pooler")
        encoder = SNGP(backbone, num_classes=num_labels, device=device)
    if model_type == 'symp':
        model_dict = torch.load("./saved/models/Symp/fold_"+str(fold)+"/model_best.pth")
        encoder = module_arch.Symp(28, 64, [1], 0.2, num_labels)
    if model_type == 'sympc':
        model_dict = torch.load("./saved/models/Symp_with_context/fold_"+str(fold)+"/model_best.pth")
        encoder = module_arch.Symp(47, 64, [1], 0.2, num_labels)
    encoder.load_state_dict(model_dict['state_dict'])
    encoder = encoder.to(device)
    encoder.eval()
    return encoder
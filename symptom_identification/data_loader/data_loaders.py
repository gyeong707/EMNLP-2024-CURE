from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
import torch
import ast
import dgl

# Default setting
class MultiLabelDataLoader(Dataset):
    def __init__(self, X, y, tokenizer, max_length):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        text = self.X[idx]
        label = torch.FloatTensor(self.y[idx])

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



#-----------------------------------External Functions------------------------------------#
def revert_label_index(labels, threshold, dictionary):
    labels = (labels > threshold) 
    data = []
    for i in range(labels.shape[0]):
        temp = []
        row = labels[i]
        indices = (row == 1).nonzero().flatten().tolist()
        for idx in indices:
            temp.append(dictionary[idx])
        data.append(temp)
    return data


def create_label_index(data, label_cols, num_labels):
    data[label_cols] = data[label_cols].map(lambda x: ast.literal_eval(x))
    labels = torch.zeros(len(data), num_labels)
    for i in range(len(data)):
        target = data[label_cols][i]
        if target == []: pass
        else:
            for t in target:
                labels[i][t] = 1
    return labels


def stratified_split(data, label, split_label, fold, seed, n_splits=5, shuffle=True):
    # This function performs stratified splitting of data into train and test sets.
    X = data['pre_question'].values  # sentence
    y = label
    y_ = split_label
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    
    X_train, X_test, y_train, y_test = None, None, None, None
    for i, (train_index, test_index) in enumerate(mskf.split(X, y_)):
        if i == fold:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            break  # Stop iteration after the required fold is found
    if X_train is None:
        raise ValueError("Fold number out of range. It should be between 0 and n_splits-1.")

    print(len(train_index), len(test_index))
    return X_train, X_test, y_train, y_test, train_index, test_index


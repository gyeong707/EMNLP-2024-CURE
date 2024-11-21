from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

def accuracy(threshold, num_labels, y_pred, y_true):
    calculator = MultilabelAccuracy(threshold=threshold, num_labels=num_labels)
    score = calculator(y_pred, y_true)
    return score

def precision(threshold, num_labels, y_pred, y_true):
    calculator = MultilabelPrecision(threshold=threshold, num_labels=num_labels)
    score = calculator(y_pred, y_true)
    return score

def recall(threshold, num_labels, y_pred, y_true):
    calculator = MultilabelRecall(threshold=threshold, num_labels=num_labels)
    score = calculator(y_pred, y_true)
    return score

def f1score(threshold, num_labels, y_pred, y_true):
    calculator = MultilabelF1Score(threshold=threshold, num_labels=num_labels)
    score = calculator(y_pred, y_true)
    return score

def print_classification_report(threshold, target_name, y_pred, y_true, output_dict=False):
    y_pred = (y_pred.detach().numpy() > threshold) 
    y_true = y_true.detach().numpy()

    report = classification_report(
      y_true,
      y_pred,
      target_names=target_name,
      zero_division=0,
      output_dict=output_dict
    )
    print(report)
    return report

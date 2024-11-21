from torchmetrics.classification import MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score
from sklearn.metrics import classification_report

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
    upper, lower = 1, 0

    report = classification_report(
      y_true,
      y_pred,
      target_names=target_name,
      zero_division=0,
      output_dict=output_dict
    )

    print(report)
    return report
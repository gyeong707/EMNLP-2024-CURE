#!/bin/bash

MODEL_NAME="BertMultiLabelClassification"
FOLD=(0 1 2 3 4)

for i in "${FOLD[@]}"; do
    echo "Running training script with FOLD: $i on $MODEL_NAME model"
    python train.py -c config_bert_sngp.json -fold $i -session "fold_$i"
    python test.py -c config_bert_sngp.json -fold $i -session "fold_$i" -ckp "saved/models/$MODEL_NAME/fold_$i/model_best.pth"
    python inference.py -c config_bert_sngp.json -fold $i -session "fold_$i" -ckp "saved/models/$MODEL_NAME/fold_$i/model_best.pth"
done

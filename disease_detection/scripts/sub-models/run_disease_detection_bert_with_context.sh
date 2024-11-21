#!/bin/bash


MODEL_NAME="bert_with_context"
SESSION_NAME="BERT_with_context"

FOLD=(1 2 3 4) # (0 1 2 3 4)

for i in "${FOLD[@]}"; do
    echo "Running training script with FOLD: $i on $MODEL_NAME model"
    python train.py -c config/sub_models/config_$MODEL_NAME.json -fold $i -session "fold_$i" 
    python test.py -c config/sub_models/config_$MODEL_NAME.json -fold $i -session "fold_$i" -ckp "saved/models/$SESSION_NAME/fold_$i/model_best.pth" 
done

python auto_report.py --model $MODEL_NAME



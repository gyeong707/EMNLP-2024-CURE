#!/bin/bash

MODEL_NAME="cure"
SESSION_NAME="CURE"
FOLD=(0) # (0 1 2 3 4)

for i in "${FOLD[@]}"; do
    echo "Running training script with FOLD: $i on $MODEL_NAME model"
     python train.py -c config/config_$MODEL_NAME.json -fold $i -session "fold_$i" -symptom "data/symptom/symptom_scores_fold_"$i".pkl" -uncertainty "data/uncertainty/uncertainty_scores_fold_"$i".pkl"
    python test.py -c config/config_$MODEL_NAME.json -fold $i -session "fold_$i" -ckp saved/models/$SESSION_NAME/fold_$i/model_best.pth -symptom "data/symptom/symptom_scores_fold_"$i".pkl" -uncertainty "data/uncertainty/uncertainty_scores_fold_"$i".pkl"
done

python auto_report.py --model $SESSION_NAME
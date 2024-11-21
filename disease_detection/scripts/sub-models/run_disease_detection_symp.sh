#!/bin/bash

MODEL_NAME="symp"
SESSION_NAME="Symp"
FOLD=(0 1 2 3 4)

for i in "${FOLD[@]}"; do
    echo "Running training script with FOLD: $i on $MODEL_NAME model"
    python train.py -c config/sub_models/config_$MODEL_NAME.json -fold $i -session "fold_$i" -symptom "data/symptom/symptom_scores_fold_"$i".pkl"
    python test.py -c config/sub_models/config_$MODEL_NAME.json -fold $i -session "fold_$i" -ckp "saved/models/"$SESSION_NAME"/fold_$i/model_best.pth" -symptom "data/symptom/symptom_scores_fold_"$i".pkl"
done

python auto_report.py --model $SESSION_NAME
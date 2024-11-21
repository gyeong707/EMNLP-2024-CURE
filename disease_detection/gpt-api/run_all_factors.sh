#!/bin/bash

# Run all factors
DATA_FILES=("data/target.csv") # "data/non_disease.csv" "data/depression.csv" "data/anxiety.csv" "data/sleep.csv" 
FACTORS=("cause") # "cause" "condition" "age" "frequency" "duration" 

for DATA in ${DATA_FILES[@]}; do
    echo "Data: $DATA"
    for FACTOR in ${FACTORS[@]}; do
        echo "Factor: $FACTOR"
        SAVE_PATH="./langchain/result/${FACTOR}/"
        python extract_factor_${FACTOR}.py --data $DATA --save_path $SAVE_PATH --factor $FACTOR
    done
done


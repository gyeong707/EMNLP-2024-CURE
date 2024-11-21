# CURE: Context and Uncertainty-Aware Mental Disorder Detection [EMNLP 2024]

This is the official implementation of "CURE: Context and Uncertainty-Aware Mental Disorder Detection" accepted in The 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP, 2024). The paper is available at [https://aclanthology.org/2024.emnlp-main.994/].

## News 
🎉 September 2024, Paper accepted at EMNLP 2024 🎉

## Requirements 
- torch==2.3.1
- tensorflow==2.14.0
- transformers==4.41.2

## Data Preparation
Our dataset KoMOS (Korean Mental Health Dataset with Mental Disorder and Symptoms labels) is publicly available in this repository. If you use this dataset, please make sure to cite our work.


## Model Architecture
CURE consists of three main components:
1. Feature Extraction
2. Model Prediction
3. Uncertainty-aware Decision Fusion

### 1. Feature Extraction
For convenience, we provide pre-extracted features in the dataset. However, if you want to extract features from scratch, follow these steps:

#### a. Symptom Identification
Navigate to the symptom identification directory and run:
```bash
bash run_symptom_identification.sh
```
This will generate symptom vectors and uncertainty scores for each fold.

#### b. Context Factor Extraction
Navigate to `disease_detection/gpt-api` and execute:
```bash
bash gpt-api/run_all_factors.sh
```
Note: Modify file paths as needed for your environment.

### 2. Model Prediction
To train and save checkpoints for the five sub-models, run the following scripts:

```bash
# Sub-models training
bash scripts/sub-models/run_disease_detection_bert.sh
bash scripts/sub-models/run_disease_detection_bert_with_context.sh
bash scripts/sub-models/run_disease_detection_symp.sh
bash scripts/sub-models/run_disease_detection_symp_with_context.sh
```

GPT results are included in the dataset for convenience. If you want to run GPT inference yourself, refer to `disease_detection/gpt-api/mental_disorder_detection.py`.

### 3. Uncertainty-aware Decision Fusion
Train the final model by running:
```bash
bash scripts/run_disease_detection_cure.sh
```


## Citation
```bibtex
@inproceedings{
  kang2024cure,
  title={CURE: Context-and Uncertainty-Aware Mental Disorder Detection},
  author={Kang, Migyeong and Choi, Goun and Jeon, Hyolim and An, Ji Hyun and Choi, Daejin and Han, Jinyoung},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages={17924--17940},
  year={2024}
}
```

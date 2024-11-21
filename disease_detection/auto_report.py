import pandas as pd
import argparse

def save_report(result, save_path, model):
    file_name = f"{model}_classification_report_all_fold.csv"
    result.to_csv(save_path+file_name, index=False)
    print("Save Report.")
    
def disease_report(disease, df_0, df_1, df_2, df_3, df_4):
    d0 = df_0[df_0['Unnamed: 0'] == disease]
    d1 = df_1[df_1['Unnamed: 0'] == disease]
    d2 = df_2[df_2['Unnamed: 0'] == disease]
    d3 = df_3[df_3['Unnamed: 0'] == disease]
    d4 = df_4[df_4['Unnamed: 0'] == disease]
    disease = pd.concat([d0, d1, d2, d3, d4], axis=0)[['precision', 'recall', 'f1-score', 'support']]
    disease.reset_index(drop=True, inplace=True)
    return disease

def auto_report_multi(model, dir_path, save_path):
    fold_0 = dir_path+"fold_0/classification_report_fold_0.csv"
    fold_1 = dir_path+"fold_1/classification_report_fold_1.csv"
    fold_2 = dir_path+"fold_2/classification_report_fold_2.csv"
    fold_3 = dir_path+"fold_3/classification_report_fold_3.csv"
    fold_4 = dir_path+"fold_4/classification_report_fold_4.csv"

    # Read CSV
    df_0 = pd.read_csv(fold_0)
    df_1 = pd.read_csv(fold_1)
    df_2 = pd.read_csv(fold_2)
    df_3 = pd.read_csv(fold_3)
    df_4 = pd.read_csv(fold_4)

    # Disease
    dep = disease_report("Depression", df_0, df_1, df_2, df_3, df_4)
    anx = disease_report("Anxiety", df_0, df_1, df_2, df_3, df_4)
    slp = disease_report("Sleep", df_0, df_1, df_2, df_3, df_4)
    eat = disease_report("Eating", df_0, df_1, df_2, df_3, df_4)
    oth = disease_report("Non-Disease", df_0, df_1, df_2, df_3, df_4)

    # Concatenate
    cat = pd.concat([dep, anx, slp, eat, oth], axis=1)

    # Average Score (for each metrics)
    precision = cat['precision'].mean(axis=1).to_frame(name='precision')
    recall = cat['recall'].mean(axis=1).to_frame(name='recall')
    f1score = cat['f1-score'].mean(axis=1).to_frame(name='f1-score')
    support = cat['support'].mean(axis=1).to_frame(name='support')

    # Concatenate
    cat = pd.concat([precision, recall, f1score, support, cat], axis=1)
    # Average Score (for each rows)
    result = pd.concat([cat, cat.mean().to_frame().transpose()], axis=0) 
    save_report(result, save_path, model)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some parameters.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model")
    args = parser.parse_args()
    
    dir_path = f"./saved/log/{args.model}/"
    save_path = f"./saved/log/{args.model}/"
    
    auto_report_multi(args.model, dir_path, save_path)
    print("Done.")
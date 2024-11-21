# import openai
import pandas as pd
import argparse
import os
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import classification_report
import torch
import ast
import numpy as np
import tqdm

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.chat_models import ChatOpenAI

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

def multilabel_stratified_split(data, label, seed, fold, n_splits=5, shuffle=True, mode='default', column_name='pre_question'):
    X = data[column_name].values
    y = label
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, test_index) in enumerate(mskf.split(X, y)):
        print(i, fold)
        if fold == i:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            return X_train, X_test, y_train, y_test, train_index, test_index
        
def read_template(fold_number):
    file_path = f"utils/gpt_template/template_fold_{fold_number}.txt"
    try:
        with open(file_path, 'r') as file:
            template = file.read()
        return template
    except FileNotFoundError:
        print(f"No template found for fold {fold_number}")
        return None
    
def convert_result(text):
    # 각 항목에 대한 초기값 설정: 모두 0으로 설정
    result = [0, 0, 0, 0, 0]
    
    # 항목별로 확인하면서 특정 항목이 나타나면 해당 위치의 값을 변경
    if "Depression: 1" in text:
        result[0] = 1
    if "Anxiety: 1" in text:
        result[1] = 1
    if "Sleep: 1" in text:
        result[2] = 1
    if "Eating: 1" in text:
        result[3] = 1
    if "Non-Disease: 1" in text:
        result[4] = 1
        
    return result

def find_indices_of_ones(input_list):
    return [i for i, value in enumerate(input_list) if value == 1]

def extract(model, num_labels):
    path = "data/ours/240523_merged_modified_factor_add_factor_text_length+6349.csv"
    data = pd.read_csv(path)
    label_index = create_label_index(data, 'disease_idx', num_labels)
    y_test = label_index
    
    api_key = "api" 
    
    template = read_template(0)
                        
    predictions = []
    predict_list = []
    save_interval = 10  # 데이터 10개마다 저장
    
    # 최종 파일 경로
    final_path = "data/ours/240523_merged_modified_factor_add_factor_text_length+6349+gpt_4o_result.csv"
    
    for i, text in enumerate(tqdm.tqdm(data['pre_question'])):
        print("question", text)
        prompt = PromptTemplate.from_template(template)
        chat = ChatOpenAI(model="gpt-4o-2024-05-13",
                        temperature=0,
                        openai_api_key = api_key)

        output_parser = StrOutputParser()
        chain =  prompt | chat | output_parser
        predicted_labels = chain.invoke({"question": text})   
        predicted_labels = convert_result(predicted_labels)
        
        print("predictions",predicted_labels)
        predictions.append(predicted_labels)
        predict_list.append(str(predicted_labels))
        
        # 데이터 10개마다 중간 저장
        if (i + 1) % save_interval == 0 or i == len(data) - 1:
            data.loc[:i, 'predicted_labels'] = predict_list
            data.to_csv(final_path, index=False)
            print(f"Saved intermediate results to {final_path}")
    
    data['predicted_labels'] = predict_list

    y_pred = predict_list
    report = classification_report(label_index, predictions, output_dict=True)
    print(f"Classification report :\n{report}")

    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.rename(index={'0': 'Depression', '1': 'Anxiety', '2': 'Sleep','3': 'Eating', '4': 'Non-Disease'})

    log_dir = 'saved/log/'+model+'/all_result'

    os.makedirs(log_dir, exist_ok=True)
    report_df.to_csv(log_dir+"/classification_report_.csv")

    X_test = data['pre_question']
    y_test = data['disease_idx']
    
    res_df = pd.DataFrame(data={'text': X_test, 'label': y_test, 'pred': y_pred})
    res_df.to_csv(log_dir+'/error_analysis.csv', index=True)
    
    # 최종 DataFrame을 CSV 파일로 저장
    data.to_csv(final_path, index=False)
    print(f"Saved final results to {final_path}")

def test(model, fold, label_cols, SEED, num_labels):
    if model =="GPT3.5":
        path = "data/ours/240523_merged_modified_factor_add_factor_text_length+6349+gpt3.5_result.csv"
    else:
        path = "data/ours/240523_merged_modified_factor_add_factor_text_length+6349+gpt4o_result.csv"
    data = pd.read_csv(path)
    
    label_index = create_label_index(data, label_cols, num_labels)
    
    X_train, X_test, y_train, y_test, train_index, test_index = multilabel_stratified_split(
        data, label_index, SEED, fold, n_splits=5, shuffle=True, mode='gpt', column_name='pre_question')
    
    X_test2 = data.loc[test_index]
    
    # 'predicted_labels' 컬럼을 리스트로 변환하여 y_pred에 할당
    y_pred = X_test2['predicted_labels'].apply(eval).tolist()
    
    # y_test와 y_pred를 비교하여 classification report 생성
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Classification report:\n{report}")
    
    # classification report를 DataFrame으로 변환 및 저장
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.rename(index={'0': 'Depression', '1': 'Anxiety', '2': 'Sleep', '3': 'Eating', '4': 'Non-Disease'})
    
    # # y_test와 y_pred의 1의 인덱스를 찾아서 저장
    y_test2 = [find_indices_of_ones(lst) for lst in y_test]
    y_pred2 = [find_indices_of_ones(lst) for lst in y_pred]
    
    # 결과 저장 디렉토리 생성
    log_dir = 'saved/log/' + model + '/fold_' + str(fold)
    os.makedirs(log_dir, exist_ok=True)
    
    # classification report를 CSV 파일로 저장
    report_df.to_csv(log_dir + "/classification_report_fold_" + str(fold) + ".csv")
    
    # 오류 분석을 위한 결과 저장
    res_df = pd.DataFrame(data={'text': X_test, 'label': y_test2, 'pred': y_pred2})
    res_df.to_csv(log_dir + '/error_analysis_fold_' + str(fold) + ".csv", index=True)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model", type=str, default="GPT-3.5", help="Model of operation")
    parser.add_argument("--fold", type=int, help="fold")
    parser.add_argument("--label_cols", type=str, default="disease_idx", help="Column name for labels")
    parser.add_argument("--SEED", type=int, default=42, help="Seed for randomness")
    parser.add_argument("--num_labels", type=int, default=5, help="number of labels")
    
    args = parser.parse_args()
    # extract(args.model, args.num_labels)
    test(args.model, args.fold, args.label_cols, args.SEED, args.num_labels)


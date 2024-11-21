from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.chat_models import ChatOpenAI
import pandas as pd
import argparse
import json
import time
import os


api_key = "api"


template = """
INSTRUCTION
I will show you a text in which an anonymous user describes their mental status. 
Based on this, add labels according to the following criteria:

Cause: Has the user recently experienced an event that could cause stress?
1: Yes. The user has recently experienced an event that could cause stress.
0: No. There is no indication of a stress-related event experienced by the user.

Let’s think step by step.
1. Identify the main symptoms the user is experiencing.
2. Identify any stressful events that might be causing those symptoms.

For objective information extraction, provide a reason to support your output.
Please answer me in Korean, and please output the result as a JSON file.

Here are some examples for you.

EXAMPLE1 
Question: 최근에 스트레스를 많이 받고 불안해하고 사람들이 다 날 보고 잇는 것 같고 제 자신이 너무 불쌍하고 난 왜 이러지 하는 상태였는데 며칠 전에도 어김없이 그런 생각이 머릿속에 꽉 찼는데 숨쉬기가 힘든 거예요 약간 심하게 울 때 헐떡헐떡 거리잖아요 그게 미세하게 계속 반복되는 느낌이었어요 
Answer: 
-Symptom: 불안감, 숨쉬기 어려움
-Cause: 1
-Cause_reason: 사용자는 최근에 스트레스를 많이 받았다고 보고함

EXAMPLE2
Question: 기억력이 갑자기 떨어진 거 같아요 예전엔 괜찮았는데 지금은 뭘 하려고 했는지 기억도 안 나고 누가 건들기만 해도 죽여버리고 싶은 생각이 들고 내가 지금 이래도 되는지 죄책감도 들어요 친구가 병원 좀 가보라 해서 글 써봐요
Answer: 
-Symptom: 기억력 저하, 죄책감, 분노감
-Cause: 0
-Cause_reason: 사용자는 갑자기 이러한 증상이 발생했다고 보고함

Now, it's your turn.
Question:{question}
Answer:
"""

def run_langchain(template, text):
    prompt = PromptTemplate.from_template(template)
    chat = ChatOpenAI(model="gpt-4o-2024-05-13",
                      temperature=0,
                      openai_api_key=api_key)

    output_parser = StrOutputParser()
    chain =  prompt | chat | output_parser
    result = chain.invoke({"question": text})     
    return result

def json_to_dataframe(json_str):
    try:
        data = json.loads(json_str)
        if type(data['Symptom']) == list:
            data['Symptom'] = ', '.join(data['Symptom'])
    except:
        data = {'Symptom': "Error", 'Cause': "Error", 'Cause_reason': "Error"}
    df = pd.DataFrame(data, index=[0])
    return df

def main(data_file, save_path, factor_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Create Directory ", save_path)

    df = pd.read_csv(data_file)
    data = df['pre_question'][3230:].reset_index(drop=True)

    result_list = []
    for i in range(len(data)):
        print('-------------', i, '-----------')
        result = run_langchain(template, data[i])
        result = result.replace('json', '').strip()
        result = result.replace('```', '').strip()
        print(data[i])
        print(result)
        result_list.append(result)

        if len(result_list) % 10== 0:
            dfs = [json_to_dataframe(result) for result in result_list]
            combined_df = pd.concat(dfs, ignore_index=True)
            now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            combined_df.to_csv(save_path+now+factor_type+"_length"+str(len(result_list))+".csv", index=False)
            print("Save...")

    dfs = [json_to_dataframe(result) for result in result_list]
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv("result/"+now+"_"+factor_type+".csv", index=False)

if __name__ == "__main__":
    print("Start Process")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Data Path")
    parser.add_argument("--save_path", type=str, help="Save Path")
    parser.add_argument("--factor", type=str, help="Factor")
    args = parser.parse_args()
    main(args.data, args.save_path, args.factor)
    print("Done")
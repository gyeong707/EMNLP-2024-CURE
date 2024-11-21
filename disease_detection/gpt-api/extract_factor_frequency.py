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

Frequency: How often do users experience psychiatric symptoms?
2: Symptoms are expressed frequently (3 or more times per week) or every day.
1: Symptoms are expressed Intermittently (less than 3 times a week) or only in specific situations. 
0: The frequency is not mentioned in the given text.  

Let’s think step by step. 
1. Identify the main symptoms the user is experiencing.
2. Determine how often the user experiences these symptoms.

For objective information extraction, provide a reason to support your output. 
If you output 1 or 2, you must provide clear evidence.
Output 0 if there are no mentions, and "언급없음" for the reason.

For objective information extraction, provide a reason to support your output.
Please answer me in Korean, and please output the result as a JSON file.

Here are some examples for you.

EXAMPLE1
Question: 지금 17살인데 아까 집 비말 번호를 까먹었는데 이 비밀번호가 몇 달 전부터 이 비밀번호였거든요 그런데 저가 이전 비밀번호를 쳤어요 왜 이런 거죠 치매가 일찍 올 수 있나요 스트레스를 받아 그런 거일까요 제가 요즘 스트레스를 많이 받기는 해요
Answer:
-Frequency: 1
-Frequency_reason: 사용자는 해당 증상을 방금 한 번 경험했습니다.

EXAMPLE2
Question: 22살 남자인데요 평소에는 밝다는 이야기 많이 듣는데 아니에요 집에 혼자 있으면 매일 울고 심하면 자해까지 합니다 스트레스받으면 자해하고 슬퍼도 자해하고 울고 갑자기 우울해져서 울고 그럽니다 어쩔 때는 이렇게까지 바닥으로 떨어진 제 자신 땜에 울고 내가 불쌍하다 생각해서 울고 맨날 혼자 소리 없이 웁니다 우울증인가요 만약 우울증이면 어디 찾아가면 되나요 정신병원은 나중에 기록 남아서 안 좋다는데
Answer
-Frequency: 2
-Frequency_reason: 사용자는 해당 증상을 매일 경험한다고 언급하고 있습니다.

Now, it's your turn.
Question: {question}
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
    except:
        data = {'Frequency': "Error", 'Frequency_reason': "Error"}
    df = pd.DataFrame(data, index=[0])
    return df

def main(data_file, save_path, factor_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(data_file)
    data = df['pre_question'][3540:].reset_index(drop=True)

    result_list = []
    for i in range(len(data)):
        print('-------------', i, '-----------')
        result = run_langchain(template, data[i])
        result = result.replace('json', '').strip()
        result = result.replace('```', '').strip()
        print(data[i])
        print(result)
        result_list.append(result)

        if len(result_list) % 10 == 0:
            dfs = [json_to_dataframe(result) for result in result_list]
            combined_df = pd.concat(dfs, ignore_index=True)
            now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
            combined_df.to_csv(save_path+now+factor_type+"_length"+str(len(result_list))+".csv", index=False)

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
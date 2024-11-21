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

Duration: Does the given text mention a period of time when the user experienced psychiatric symptoms?
2: The user has been experiencing symptoms for a month or more. This is typically expressed in the text as "1년 째," "한달 넘게," "오래 전부터," etc.
1: The user has been experiencing symptoms for less than a month. This is typically expressed in the text as "요즘들어," "최근에," "일주일," etc.
0: The duration is not mentioned in the given text.

Age: Does the text mention the user's age or age range?
2: The user is in their 20s or older, which means the user is an adult and is referred to in the text as “성인,” “대학생,” “20대,” “주부,” etc.
1: The user is a teenager or younger, which means the user is a child or young adult and is mentioned in the text as “중학생,” “고등학생,” “초등학생,” “고3,” “중2,” or a specific age.
0: There is no mention of age in the text.

Let’s think step by step.
1. Identify the symptoms the user is experiencing.
2. Determine how long the user has experienced these symptoms.
3. Determine user's age.

For objective information extraction, find and present evidence from the text to support your output. 
If you output 1 or 2, you must provide clear evidence from the text.
Output 0 if there are no mentions, and "언급없음" for the reason.

Please answer me in Korean, and please output the result as a JSON file.
Here are some examples for you.

EXAMPLE1
Question: 버스에서 사람이 많으면 땀에 머리카락이 젖고 이마에서 땀방울이 흐르고 등줄기에서 땀이 흘러내리고 손에 식은땀이 나고 행동도 어정쩡하게 되는데 왜 이러는 건가요
Answer: 
-Duration: 0
-Duration_reason: 언급없음
-Age: 0
-Age_reason: 언급없음

EXAMPLE2
Question: 안녕하세요 고2 남학생입니다 제가 1달 전에 밥 먹고 장염이 걸려서 심하게 앓았었는데요 그때 이후부터 모든 음식만 보면 배가 살살 아프고 속이 매스꺼워서 도저히 정상적인 식사를 할 수 없습니다 아침엔 밥 13 정도만 먹고 학교에서 점심은 아예 안 먹고 저녁도 거의 반 공기 정도 밖에 안 먹는 것 같습니다 하루에 먹는 양이 13 정도로 줄어서 원래부터 마른 체형이었는데 점점 더 말라가고 있습니다 한 달 전에는 181cm에 53kg이었는데 지금은 48kg밖에 안 나갑니다 살이 더 빠지면 안 될 것 같아서 꾸역꾸역 먹어보려고도 노력했지만 그때마다 구역질이 나오고 요즘은 어지러움까지 느껴서 매번 실패했습니다 제가 거식증에 대해 찾아보니까 거식증은 자신이 뚱뚱하다는 것에 대한 스트레스로 음식을 거부하는 거라던데 저의 경우는 좀 다른 것 같아서 질문 올려봅니다
Answer:
-Duration: 2
-Duration_reason: 1달 전
-Age: 1
-Age_reason: 고2

EXAMPLE3
Question: 21살 여자입니다 요 며칠 계속 피곤하고 무기력해지고 아무것도 하기 싫어지고 뭘 먹으면 소화도 잘 안되고 속이 답답하고 하루 종일 정신이 멍해요 집중도 잘 안되고 밖에 나가면 시선을 어디에 둬야 될지도 모르겠고 잠을 자면서도 작은 소리 하나에 눈 떠지고 밤에 일찍 자도 하루 24시간 내내 피곤해요 또 제일 불편한 건 눈이 계속 건조하고 침침해요 그리고 비현실감도 들어요 이게 공황장애 증상인가요
Answer: 
-Duration: 1
-Duration_reason: 요 며칠
-Age: 2
-Age_reason: 21살

Now, it's your turn.
Question:{question}
Answer:
"""

def run_langchain(template, text):
    prompt = PromptTemplate.from_template(template)
    chat = ChatOpenAI(model="gpt-4o-2024-05-13",
                      temperature=0,
                      openai_api_key = api_key)

    output_parser = StrOutputParser()
    chain =  prompt | chat | output_parser
    result = chain.invoke({"question": text})     
    return result

def json_to_dataframe(json_str):
    try:
        data = json.loads(json_str)
    except:
        data = {'Duration': "Error", 'Duration_reason': "Error", "Age": "Error", "Age_reason": "Error"}
    df = pd.DataFrame(data, index=[0])
    return df

def main(data_file, save_path, factor_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df = pd.read_csv(data_file)
    data = df['pre_question']

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
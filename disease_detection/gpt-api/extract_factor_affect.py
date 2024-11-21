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

1. Symptom: What psychiatric symptoms or issues is the user experiencing?
2. Social affect: Does the symptom the user is experiencing interfere with the user's social functioning?
3. Educational affect: Does the symptom the user is experiencing interfere with the user's educational functioning?
4. Occupational affect: Does the symptom the user is experiencing interfere with the user's occupational functioning?
5. Life-threatening affect: Are the user's symptoms life-threatening?

For each affect:

1: The user's symptoms are currently affecting their social, educational, or occupational functioning, or are life-threatening.
Examples:
- Social affect: The user may experience interpersonal disconnection due to symptoms.
- Educational affect: If the user is a student or examinee, they may report that their symptoms are preventing them from focusing on their studies.
- Occupational affect: The user may report that they are unable to focus on work or have decided to leave their jobs because of their symptoms.
- Life-threatening affect: The user has made a specific suicide plan or attempted suicide, not just thoughts of suicide, or is in a severely poor health condition that poses a significant risk to their life.
0: There is no mention of educational, occupational, social, or life-threatening effects in the given text.

Let's think about it in steps.
1. Identify the symptoms the user is experiencing.
2. Determine how those symptoms have consequently affected their interpersonal, work, or academic life, or if they are life-threatening.

For objective information extraction, provide a reason to support your output. 
If you output 1, you must provide clear evidence from the text. 
Output 0 if there are no mentions, and "언급없음" for the reason.

Please answer me in Korean, and please output the result as a JSON file.
Here are some examples for you.

EXAMPLE1
Question: 피나 수술장면을 보면 식은땀과 구토감 증상이 있는데 정작 생각으론 아무렇지도 않는데 몸이 반응하네요 현기증 나고 토할 것 같고 누가 봐도 아파 보일 만큼 창백하고 식은땀 나오고 그런데 제가 그런 걸 계속 봐야 하는 직업이라 정신과 가서 약 좀 먹으면 나을까요
Answer: 
- Symptom: 식은땀, 구토감, 현기증, 창백함
- Social_affect: 0
- Social_affect_reason: 언급없음
- Educational_affect: 0
- Educational_affect_reason: 언급없음
- Occupational_affect: 1
- Occupational_affect_reason: 사용자는 피나 수술장면을 보면 식은땀과 구토감 증상이 있는데 계속 이를 봐야하는 직업을 가지고 있음
- Life-threatening_affect: 0
- Life-threatening_affect_reason: 언급없음

EXAMPLE2
Question: 고3인데요 고2까지만 해도 심각한 외향적 성격이었는데 점점 몸도 아프고 밖에 나가려고 외출복을 입는 순간부터 몸이 무거워져서 나가기가 너무 힘들어요 밖에 나가면 몸도 더 아프고 스트레스만 쌓이고 그렇다고 학교랑 대인관계 때문에 안 나갈 수가 없고 어떡하면 나가는 게 안 힘들고 괜찮아질까요 원인도 잘 모르겠네요 남의 시선 신경 쓰느라 그런 건 아니에요
Answer: 
- Symptom: 무기력, 외출 시 몸이 무거워짐
- Social_affect: 1
- Social_affect_reason: 사용자는 사회적 활동을 위한 외출이 어려운 상태에 놓여 있음. 
- Educational_affect: 0
- Educational_affect_reason: 언급없음
- Occupational_affect: 0
- Occupational_affect_reason: 언급없음
- Life-threatening_affect: 0
- Life-threatening_affect_reason: 언급없음

EXAMPLE3
Question: 수능 일주일 남았는데 저 시험 볼 때 강박증인지 공포증인지 있는 것 같아요 제가 중3 때 시험에 대한 강박사고가 심했는데 그때 시험 보다가 자주 긴장을 했어요 근데 그때 화장실 가고 싶은 기분이 자꾸 들고 또 그 기분이 설사로 변해서 시험 보다가 1시간 동안 식은땀 뻘뻘 나고 심장 쿵쾅 쿵쾅 아 시험지 내고 화장실 가면 시험 망하는데 근데 여기서 설사는 못 참으니까 싸면 이런 생각 때문에 시험이 공포로 다가와 요 시험에 대한 배변 관련 공포증이 있나 봐요 시험 때 과민할 정도로 변을 봤나 잔변감이 있나 설사약을 미리 먹고 갈까 벽에 대해 집착해요
Answer: 
- Symptom: 설사, 식은땀, 가슴 두근거림
- Social_affect: 0
- Social_affect_reason: 언급없음
- Educational_affect: 1
- Educational_affect_reason: 사용자는 현재 수능이 일주일 남은 상태이며, 시험 볼 때 강박증이나 공포증이 있는 것 같다고 호소하고 있음
- Occupational_affect: 0
- Occupational_affect_reason: 언급없음
- Life-threatening_affect: 0
- Life-threatening_affect_reason: 언급없음

EXAMPLE4
Question: 삶에 대한 의욕이 없습니다 좋아하는 프로그램을 봐도 웃어지지가 않고요 얼마 전 제가 우울증 테스트를 했습니다 의사와 상담받아야 할 심각한 우울증이라 그러더군요 자살시도 몇 번 해보려 했습니다 락스에 물 타서 먹어보려 했는데 전 고통스럽게 죽긴 싫어서 안 했고요 칼로 동맥 그어서 죽어보려 했는데 살짝 그었더니 정말 쓰라리고 아프더군요 정말 삶에 대한 의욕이 없습니다 저한테 조언 좀 해주실 분 없나요 저 좀 제발 살려주세요 저도 이제 제가 무서워요 앞에 있는 칼 보면 제가 언제 저를 벨 지 몰라서 칼이란 칼 다 치워야 돼요 누가 저 쫓아와서 죽일까 봐 밤에 잠도 혼자 못 자고 다 불 켜고 잡니다 정말 무서워요 제발 도와주세요 여러분들
Answer: 
- Symptom: 흥미저하, 우울감, 자살시도, 자살충동
- Social_affect: 0
- Social_affect_reason: 언급없음
- Educational_affect: 0
- Educational_affect_reason: 언급없음
- Occupational_affect: 0
- Occupational_affect_reason: 언급없음
- Life-threatening_affect: 1
- Life-threatening_affect_reason: 사용자는 과거에 자살시도를 했으며 현재도 자살충동을 자주 느끼는 상태임

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
        if type(data['Symptom']) == list:
            data['Symptom'] = ', '.join(data['Symptom'])
    except:
        data = {'Symptom': 'Error', 'Social_affect': 'Error', 'Social_affect_reason': 'Error', 'Educational_affect': 'Error', 'Educational_affect_reason': 'Error', 'Occupational_affect': 'Error', 'Occupational_affect_reason': 'Error', 'Life-threatening_affect': 'Error', 'Life-threatening_affect_reason': 'Error'}
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
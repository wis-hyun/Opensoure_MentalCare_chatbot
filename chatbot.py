from io import BytesIO
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time
from mental_health_chatbot.chatbot_logic import emergency_link, survey
import os
import base64
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics

CONFIG = {}
DATASET_PATH = ""


def load_config():
    global CONFIG, DATASET_PATH
    with open('config.json', 'r') as f:
        CONFIG = json.load(f)
    DATASET_PATH = CONFIG['dataset_path']


load_config()

# MP3 파일 경로 (Streamlit 앱 내부)
audio_file_path = os.path.join(DATASET_PATH, 'thema.mp3')
audio_file = open(audio_file_path, "rb")

# 오디오 플레이어 위젯 생성
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 10px;">
        <p style="font-size: 18px; color: black;">"이 플레이어로 편안한 마음을 드릴게요🍀"</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.audio(audio_file.read(), format="audio/mp3")


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: 67%; /* Adjust the percentage as needed */
            background-position: south; /* Optional: south the image */
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# 배경 이미지 추가
img_file_path = os.path.join(DATASET_PATH, 'home.png')
add_bg_from_local(img_file_path)  # 이미지 파일 이름을 정확히 입력


@st.cache_resource
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


@st.cache_data
def get_dataset():
    csv_file_path = os.path.join(DATASET_PATH, 'mental_health_chatbot', 'wellness_dataset.csv')
    df = pd.read_csv(csv_file_path)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


@st.cache_resource
def get_sentiment_model():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=3)
    return tokenizer, model


model = cached_model()
df = get_dataset()
tokenizer, sentiment_model = get_sentiment_model()

st.header('❄️ 눈송이 챗봇 ❄️')
st.markdown("[❤️하이브리드샘이솟아](https://github.com/wis-hyun/Opensoure_MentalCare_chatbot)")
st.markdown(
    f"""
    <style>
    [data-testid="stForm"] {{
        background-color: lightblue;
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'survey_mode' not in st.session_state:
    st.session_state['survey_mode'] = False

if 'survey_idx' not in st.session_state:
    st.session_state['survey_idx'] = 0

if 'survey_end' not in st.session_state:
    st.session_state['survey_end'] = False

if 'survey_type' not in st.session_state:
    st.session_state['survey_type'] = ""

if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = ""

if 'questions' not in st.session_state:
    st.session_state['questions'] = []


def reset_session():
    st.session_state['questions'] = []
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state.conversation_summary = ""  # 요약 결과 초기화
    st.session_state['survey_type'] = ""
    st.session_state['survey_idx'] = 0
    st.session_state['survey_mode'] = False
    st.session_state['survey_end'] = False

def test(test_type):
    st.session_state['survey_mode'] = True
    st.session_state['survey_end'] = False
    st.session_state['survey_type'] = test_type

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 눈송이 🩵 : ', '')
    submitted = st.form_submit_button('전송하기')

if st.session_state['survey_mode']:
    if st.session_state['survey_type'] == 'depression':
        st.session_state['questions'] = [
            "기분이 가라앉거나, 우울하거나, 희망이 없다고 느꼈다.\n&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "평소 하던 일에 대한 흥미가 없어지거나 즐거움을 느끼지 못했다.\n&emsp;&emsp;&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "잠들기가 어렵거나 자주 깼다/혹은 너무 많이 잤다.\n&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "평소보다 식욕이 줄었다/혹은 평소보다 많이 먹었다.\n&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)"
        ]

    if st.session_state['survey_type'] == 'stress':
        st.session_state['questions'] = [
            "최근 1개월 동안, 예상치 못했던 일 때문에 당황했던 적이 얼마나 있었습니까?\n&emsp;&emsp;&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "최근 1개월 동안, 인생에서 중요한 일들을 조절할 수 없다는 느낌을 얼마나 경험하였습니까?\n&emsp;&emsp;&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "최근 1개월 동안, 신경이 예민해지고 스트레스를 받고 있다는 느낌을 얼마나 경험하였습니까?\n&emsp;&emsp;&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)",
            "최근 1개월 동안, 당신의 개인적 문제들을 다루는데 있어서 얼마나 자주 자신감을 느끼셨습니까?\n&emsp;&emsp;&emsp;&emsp;&emsp;(1:없음, 2:거의 없음, 3:많음, 4:매우 많음)"
        ]

    if (st.session_state['survey_idx'] < len(st.session_state['questions'])):
        st.session_state.generated.append(st.session_state['questions'][st.session_state['survey_idx']])
        st.session_state['survey_idx'] += 1

if submitted and user_input:
    with st.spinner('처리 중...'):
        start_time = time.time()

        if st.session_state['survey_mode']:
            st.session_state.past.append(f"{user_input}")

        elif '자살' in user_input or '죽음' in user_input:
            answer = emergency_link()
            st.session_state.past.append(f"{user_input}")
            st.session_state.generated.append(answer['챗봇'])

        else:
            embedding = model.encode(user_input)
            df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
            answer = df.loc[df['distance'].idxmax()]

            # 감정 분석
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = sentiment_model(**inputs)
            sentiment_score = torch.nn.functional.softmax(outputs.logits, dim=1)
            sentiment_label_idx = torch.argmax(sentiment_score, dim=1).item()

            if sentiment_label_idx == 0:
                sentiment_label = "부정적 😢"
            elif sentiment_label_idx == 1:
                sentiment_label = "중립적 😐"
            else:
                sentiment_label = "긍정적 😊"

            st.session_state.past.append(f"{user_input} (감정: {sentiment_label})")
            st.session_state.generated.append(answer['챗봇'])

        response_time = time.time() - start_time
        st.success(f"응답 시간: {response_time:.2f}초")


# 대화 요약 함수 정의
def summarize_conversation(past, generated):
    summary = ""
    for i in range(len(past)):
        summary += f"눈송이: {past[i]}\n"  # 사용자 입력 기록 추가
        summary += f"챗봇: {generated[i]}\n\n"  # 챗봇 응답 요약
    return summary


# 대화 로그 버튼 추가
def summary_button():
    conversation_summary = summarize_conversation(st.session_state.past, st.session_state.generated)
    st.session_state.conversation_summary = conversation_summary  # 요약 결과를 세션 상태에 저장
    # st.text_area("대화 로그", value=conversation_summary, height=300)   # 대화 요약 결과를 출력할 텍스트 영역 추가


# 대화 로그 저장 기능 추가 (PDF 파일)
def summary_save():
    conversation_log = ""
    for i in range(len(st.session_state['past'])):
        conversation_log += f"눈송이: {st.session_state['past'][i]}\n"
        conversation_log += f"챗봇: {st.session_state['generated'][i]}\n\n"

    # PDF 파일로 저장
    pdf_buffer = BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)

    # 맑은고딕 폰트 등록
    pdfmetrics.registerFont(TTFont('MalgunGothic', 'malgun.ttf'))
    pdf.setFont("MalgunGothic", 12)

    y_position = 750  # 시작 위치
    for line in conversation_log.split('\n'):
        pdf.drawString(30, y_position, line)
        y_position -= 15
        if y_position < 50:
            pdf.showPage()
            pdf.setFont("MalgunGothic", 12)
            y_position = 750

    pdf.save()

    pdf_buffer.seek(0)

    # 파일 다운로드 링크 제공
    st.download_button('대화 로그 다운로드', pdf_buffer, file_name='conversation_log.pdf', mime='application/pdf')


# 버튼 배치
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.button("대화 초기화", on_click=reset_session)

with col2:
    st.button("우울증검사", on_click=lambda: test('depression'))

with col3:
    st.button("스트레스검사", on_click=lambda: test('stress'))

with col4:
    st.button('대화 로그', on_click=summary_button)

with col5:
    st.button('대화 로그 저장', on_click=summary_save)

if st.session_state.conversation_summary:
    st.text_area("대화 로그", value=st.session_state.conversation_summary, height=300)

# 대화 로그 출력
if (st.session_state['survey_mode']):
    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i) + '_bot2')  # 챗봇 응답 메시지 출력
        if len(st.session_state['past']) > i:
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user2')  # 사용자 입력 메시지 출력

    if len(st.session_state['questions']) != 0 and st.session_state['survey_idx'] == len(st.session_state['questions']):
        sum = 0
        for i in range(len(st.session_state['past'])):
            sum += int(st.session_state['past'][i])

        survey_type = ''
        if(st.session_state['survey_type'] == 'stress'):
            survey_type = '스트레스'
        else: 
            survey_type = '우울증'

        status = ''
        if (sum <= 4):
            status = '우울아님'
        elif (sum <= 8):
            status = '가벼운 우울'
        elif (sum <= 12):
            status = '중간정도의 우울\n(가까운 지역센터나 전문기관 방문을 요망합니다.)'
        elif (sum <= 16):
            status = '심한 우울\n(전문기관의 치료적 개입과 평가가 필요합니다.)'

        result = '#### 귀하의 ' + survey_type + ' 척도 테스트결과 점수는 ' + str(sum) + '점 입니다.'
        result += '\n' + status

        message(result, key=str('result') + '_result_bot')  # 챗봇 응답 메시지 출력
else:
    for i in range(len(st.session_state['past'])):
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # 사용자 입력 메시지 출력
        if len(st.session_state['generated']) > i:
            message(st.session_state['generated'][i], key=str(i) + '_bot')  # 챗봇 응답 메시지 출력

st.markdown(
    f"""
    <style>
    [data-testid="stForm"] {{
        background-color: lightblue; 
        padding: 20px; 
        border-radius: 10px; 
    }}
    .stMessage {{
        background-color: #f1f1f1; 
        border-radius: 10px; 
        padding: 10px; 
        margin-bottom: 10px; 
    }}
    .stMessage.is_user {{
        background-color: #daf7a6; 
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

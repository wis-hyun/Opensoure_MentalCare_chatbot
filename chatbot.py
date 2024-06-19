import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import time

@st.cache_resource
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache_data
def get_dataset():
    df = pd.read_csv('/Users/sunghyunkim/Desktop/mental-health-chatbot/wellness_dataset.csv')
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

# 대화 요약 함수 정의
def summarize_conversation(past, generated):
    summary = ""
    for i in range(len(past)):
        summary += f"눈송이: {past[i]}\n"   # 사용자 입력 기록 추가
        summary += f"챗봇: {generated[i]}\n\n"  # 챗봇 응답 요약
    return summary

st.header('❄️ 눈송이 챗봇 ❄️')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

def reset_session():
    st.session_state['generated'] = []
    st.session_state['past'] = []

st.button("대화 초기화", on_click=reset_session)

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 눈송이 🩵 : ', '')
    submitted = st.form_submit_button('전송하기')

if submitted and user_input:
    with st.spinner('처리 중...'):
        start_time = time.time()
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

# 대화 로그 버튼 추가
if st.button('대화 로그', key='summary_button'):
    conversation_summary = summarize_conversation(st.session_state.past, st.session_state.generated)
    st.text_area("대화 로그", value=conversation_summary, height=300)   # 대화 요약 결과를 출력할 텍스트 영역 추가

# 대화 로그 출력
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')    # 사용자 입력 메시지 출력
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

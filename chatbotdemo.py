import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Streamlit 애플리케이션 설정
st.set_page_config(layout="wide")

# 챗봇 모델과 데이터셋 로드
@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

# 대화 요약 함수 정의
def summarize_conversation(past, generated):
    summary = ""
    for i in range(len(past)):
        summary += f"눈송이: {past[i]}\n"
        summary += f"챗봇: {generated[i]}\n\n"
    return summary

# Streamlit 애플리케이션 UI 구성
st.header('눈송이 멘탈케어 챗봇')
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

# 대화 기록 초기화
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# 대화 입력 폼 및 전송 버튼 추가
with st.form('form', clear_on_submit=True):
    user_input = st.text_input('눈송이: ', '')
    submitted = st.form_submit_button('전송')

# 대화 처리 및 출력
if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

# 대화 로그 버튼 추가
if st.button('대화 로그', key='summary_button'):
    conversation_summary = summarize_conversation(st.session_state.past, st.session_state.generated)
    st.text_area("대화 로그", value=conversation_summary, height=300)

# 대화 로그 출력
for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')

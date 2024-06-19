import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import base64
import tempfile



# MP3 파일 경로 (Streamlit 앱 내부)
audio_file = open("thema.mp3", "rb")

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
add_bg_from_local('home.png')  # 이미지 파일 이름을 정확히 입력

@st.cache_resource
def cached_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')

@st.cache_resource
def get_dataset():
    file_path = "wellness_dataset.csv"
    df = pd.read_csv(file_path)
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

# Load the model and dataset
model = cached_model()
df = get_dataset()

# Verify image paths
user_image_path = 'user.png'
bot_image_path = 'profile.png'

st.header('눈송이 멘탈케어 챗봇')
st.markdown("[❤️하이브리드 샘이솟아](https://github.com/2eueu/mental-health-chatbot)")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('눈송이: ', '')
    submitted = st.form_submit_button('전송')

if submitted and user_input:
    embedding = model.encode(user_input)
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]
    answer_text = answer['챗봇']

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer_text)

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=f"user_message_{i}")
    if len(st.session_state['generated']) > i:
        if '<img src="' in st.session_state['generated'][i]:
            st.markdown(st.session_state['generated'][i], unsafe_allow_html=True)
        else:
            message(st.session_state['generated'][i], key=f"bot_message_{i}")

# Custom CSS to style the form
st.markdown(
    """
    <style>
    [data-testid="stForm"] {
        background-color: lightblue;
        padding: 20px; 
        border-radius: 10px; 
    }
    


    }
    </style>
    """,
    unsafe_allow_html=True,
)

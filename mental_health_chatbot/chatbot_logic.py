import streamlit as st
from streamlit_chat import message

def emergency_link():
    answer = '#### 당신은 소중한 사람입니다. 포기하지마세요!\n'
    answer += '🩵 [우울증과 우울감의 차이는?](https://kin.naver.com/open100/detail.nhn?d1id=7&dirId=70109&docId=1494755)\n'
    answer += '🩵 [우울을 예방 및 극복하는 방법은?](https://kin.naver.com/open100/detail.nhn?d1id=7&dirId=70109&docId=1494757)\n'
    answer += '🩵 [먼저 병원에 가야 하는 이유](https://kin.naver.com/open100/detail.nhn?d1id=7&dirId=70109&docId=1494766)\n'
    answer += '🩵 [그래도, 삶은 희망이다](https://kin.naver.com/open100/detail.nhn?d1id=7&dirId=70109&docId=1494765)\n'

    result = {'챗봇': answer}
    return result


def survey():
    print("안녕하세요! 간단한 설문조사 챗봇입니다.")

    questions = [
        "당신의 이름은 무엇인가요?",
        "당신의 나이는 몇 살인가요?",
        "당신이 좋아하는 색깔은 무엇인가요?",
        "당신이 가장 좋아하는 음식은 무엇인가요?"
    ]
    st.session_state.generated.append(questions[0])

    answers = {}

    # with st.form('form', clear_on_submit=True):
    #     user_input = st.text_input('사용자 눈송이 🩵 : ', '')
    #     submitted = st.form_submit_button('전송하기')

    # for i, question in enumerate(questions):
    #     answer = {'챗봇': question}
    #     st.session_state.generated.append(answer['챗봇'])
    #
    #     if submitted and user_input:
    #         st.session_state.past.append(f"{answer}")


    for i in range(len(st.session_state['generated'])):
        message(st.session_state['generated'][i], key=str(i) + '_bot')  # 챗봇 응답 메시지 출력
        # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')  # 사용자 입력 메시지 출력

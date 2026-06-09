from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
import re

# .env 파일에서 환경 변수 로드
load_dotenv()

def get_youtube_video_id(url: str):
    """유튜브 URL에서 Video ID만 깔끔하게 추출합니다 (Shorts 등 다양한 포맷 지원)."""
    # 정규 표현식을 사용하여 11자리 Video ID 추출
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    return None

def summarize_youtube_video(url: str):
    """
    pytube 없이 youtube-transcript-api만 사용하여 자막을 가져와 요약합니다.
    """
    video_id = get_youtube_video_id(url)
    if not video_id:
        st.error("올바른 유튜브 URL 형식이 아닙니다.")
        return None

    # 1. 자막 추출 (pytube 없이 직접 API 호출)
    try:
        # 한국어 우선, 없으면 영어 자막 가져오기
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id, languages=['ko', 'en'])
        
        # 자막 객체 리스트를 하나의 긴 텍스트로 합치기
        transcript_text = " ".join([item.text for item in transcript])
        
    except AttributeError as e:
        st.error(f"자막 라이브러리 사용 중 오류가 발생했습니다. 최신 버전 코드가 적용되었는지 확인하세요: {e}")
        return None
    except Exception as e:
        if "Subtitles are disabled" in str(e) or "No transcripts were found" in str(e):
            st.warning("이 영상에는 추출할 수 있는 자막이 존재하지 않습니다.")
        else:
            st.error(f"자막을 가져오는 중 오류가 발생했습니다: {e}")
        return None

    # 2. LLM 모델 설정
    chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

    # 3. 프롬프트 템플릿 설정 (메타데이터 제외하고 내용에만 집중)
    prompt = PromptTemplate.from_template(
        """
        당신은 뛰어난 전문 요약가입니다. 
        다음 유튜브 영상의 자막 원본을 바탕으로, 내용을 한국어로 알기 쉽게 요약해 주세요.

        [영상 자막 원본]
        {text}

        [출력 형식 가이드]
        1. 📝 **세 줄 요약**: 영상의 가장 핵심적인 내용을 3줄로 명확하게 요약하세요.
        2. 💡 **주요 내용 정리**: 상세한 주요 포인트를 불릿 포인트(-)를 사용하여 정리하세요.
        3. 🎯 **결론**: 이 영상이 전달하고자 하는 최종 메시지를 작성하세요.
        """
    )

    # 4. 체인 구성 및 실행
    chain = prompt | chat_model
    
    try:
        # 추출한 자막 텍스트만 LLM에 전달
        response = chain.invoke({"text": transcript_text})
        return response.content
    except Exception as e:
        st.error(f"AI 요약 중 오류가 발생했습니다: {e}")
        return None

# --- Streamlit UI ---
st.title("🎬 유튜브 영상 요약기 (안정화 버전)")
st.write("말썽 많은 pytube를 제거했습니다. 유튜브 링크를 입력하면 AI가 영상 자막을 분석하여 요약합니다.")

url_input = st.text_input("여기에 유튜브 URL을 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("요약 시작!"):
    if url_input:
        with st.spinner("자막을 추출하고 요약 중입니다... 잠시만 기다려주세요."):
            summary = summarize_youtube_video(url_input)
            if summary:
                st.subheader("📊 AI 요약 결과")
                st.markdown(summary)
    else:
        st.warning("요약할 유튜브 영상의 URL을 입력해주세요.")
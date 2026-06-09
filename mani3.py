from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate
import urllib.error

# .env 파일에서 환경 변수 로드
load_dotenv()

def summarize_youtube_video(url: str):
    """
    주어진 유튜브 URL의 자막을 가져와 요약합니다.
    """
    try:
        # 1. 유튜브 로더를 사용하여 자막 및 메타데이터 로드
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["ko", "en"]  # 한국어 우선, 없으면 영어
        )
        docs = loader.load()
    except urllib.error.HTTPError as e:
        if e.code == 400:
            st.error("유튜브 영상 정보를 가져오는데 실패했습니다 (HTTP 400). 'pytube' 라이브러리 버전 문제일 수 있습니다. `pip install --upgrade pytube`로 업데이트 해보세요.")
        else:
            st.error(f"영상 정보나 자막을 가져오는데 실패했습니다 (HTTP {e.code}):\n{e}")
        return None
    except Exception as e:
        if 'Could not import "pytube"' in str(e):
            st.error("유튜브 영상 정보 로딩에 필요한 'pytube' 라이브러리가 설치되지 않았습니다. `pip install pytube` 명령어로 설치 후 다시 시도해 주세요.")
        elif 'Could not import "youtube_transcript_api"' in str(e):
            st.error("자막 추출에 필요한 'youtube-transcript-api' 라이브러리가 설치되지 않았습니다. `pip install youtube-transcript-api` 명령어로 설치 후 다시 시도해 주세요.")
        elif "HTTPException" in str(e) or "RegexMatchError" in str(e):
             st.error("영상을 처리하는 중 'pytube' 라이브러리에서 오류가 발생했습니다. 라이브러리 버전 문제일 가능성이 높습니다. `pip install --upgrade pytube` 명령어로 업데이트를 시도해 보세요.")
        else:
            st.error(f"영상 정보를 가져오는 중 예상치 못한 오류가 발생했습니다: {e}")
        return None

    if not docs:
        st.warning("이 영상에서는 추출할 수 있는 자막을 찾지 못했습니다.")
        return None

    content = docs[0].page_content
    metadata = docs[0].metadata
    title = metadata.get('title', '제목 없음')
    author = metadata.get('author', '알 수 없는 채널')

    # 2. LLM 모델 설정
    chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

    # 3. 프롬프트 템플릿 설정 (mani2.py의 프롬프트를 활용)
    prompt = PromptTemplate.from_template(
        """
        당신은 뛰어난 전문 요약가입니다. 
        다음 유튜브 영상의 자막을 바탕으로, 내용을 한국어로 알기 쉽게 요약해 주세요.

        [영상 정보]
        - 제목: {title}
        - 채널명: {author}

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
        response = chain.invoke({"title": title, "author": author, "text": content})
        return response.content
    except Exception as e:
        st.error(f"AI 요약 중 오류가 발생했습니다: {e}")
        return None

# --- Streamlit UI ---
st.title("🎬 유튜브 영상 요약기")
st.write("유튜브 링크를 입력하면 AI가 영상 내용을 분석하여 요약해 드립니다.")

url_input = st.text_input("여기에 유튜브 URL을 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("요약 시작!"):
    if url_input:
        with st.spinner("영상을 분석하고 요약 중입니다... 잠시만 기다려주세요."):
            summary = summarize_youtube_video(url_input)
            if summary:
                st.subheader("📊 AI 요약 결과")
                st.markdown(summary) # 마크다운 형식으로 출력하여 가독성 향상
    else:
        st.warning("요약할 유튜브 영상의 URL을 입력해주세요.")
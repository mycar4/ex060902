import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- UI 설정 ---
st.set_page_config(page_title="유튜브 요약 & 번역기", page_icon="🎥")
st.title("🎥 유튜브 링크만 넣으면 끝! 영상 요약 및 번역기")
st.markdown("유튜브 URL을 입력하면, 영상의 자막을 추출하여 핵심 내용을 요약하고 한국어로 번역해 줍니다.")

# --- 메인 화면: URL 입력 ---
youtube_url = st.text_input("유튜브 영상 링크를 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

if st.button("요약 및 번역 시작"):
    # 소스(.env)에 API 키가 잘 들어있는지 확인
    if not openai_api_key:
        st.error("⚠️ 서버에 OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        st.stop()
        
    if not youtube_url:
        st.warning("⚠️ 유튜브 링크를 입력해 주세요.")
        st.stop()

    with st.spinner("자막을 추출하고 요약하는 중입니다... 잠시만 기다려주세요! ⏳"):
        try:
            # 1. 유튜브 자막 로드 (400 에러 방지를 위해 add_video_info=False 유지)
            loader = YoutubeLoader.from_youtube_url(
                youtube_url, 
                add_video_info=False,
                language=["en", "ko", "ja", "es"]
            )
            docs = loader.load()
            
            if not docs:
                st.error("자막을 찾을 수 없는 영상입니다. (자동 생성 자막이 꺼져있을 수 있습니다)")
                st.stop()
                
            transcript = docs[0].page_content
            video_title = docs[0].metadata.get("title", "제목 없음")
            
            # 2. LangChain 프롬프트 및 LLM 설정
            llm = ChatOpenAI(
                temperature=0, 
                openai_api_key=openai_api_key, 
                model_name="gpt-3.5-turbo"
            )
            
            prompt = PromptTemplate.from_template(
                """
                당신은 전문 번역가이자 요약 전문가입니다. 
                다음은 유튜브 영상의 전체 자막입니다. 
                이 내용을 꼼꼼히 읽고, 아래 양식에 맞춰 한국어로 명확하게 작성해 주세요.
                
                1. 영상의 핵심 주제 (1줄)
                2. 상세 요약 (3~5개의 글머리 기호 사용)
                
                자막 내용:
                {transcript}
                """
            )
            
            # 3. 체인 생성 및 실행 (LCEL 방식)
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"transcript": transcript})
            
            # 4. 결과 출력
            st.success("✅ 요약이 완료되었습니다!")
            st.subheader(f"📺 영상 제목: {video_title}")
            st.markdown("---")
            st.markdown(result)
            
            with st.expander("원본 자막 내용 보기"):
                st.text(transcript)
                
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.info("올바른 유튜브 링크인지, 또는 해당 영상이 자막을 제공하는지 확인해 주세요.")
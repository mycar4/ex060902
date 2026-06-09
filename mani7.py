import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# .env 파일에서 환경 변수(API 키) 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- UI 설정 ---
st.set_page_config(page_title="맞춤형 마케팅 글 생성기", page_icon="✍️", layout="wide")
st.title("✍️ 플랫폼 맞춤형 마케팅 카피라이터")
st.markdown("상품 정보와 타겟 고객을 입력하면, AI가 선택한 SNS 채널과 톤앤매너에 최적화된 마케팅 문구를 작성해 줍니다.")

# --- 메인 화면: 입력 폼 ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 🎯 어떤 상품/서비스인가요?")
        product_name = st.text_input("상품/서비스 이름", placeholder="예: 무소음 무선 마우스 로지텍 M350")
        target_audience = st.text_input("타겟 고객", placeholder="예: 카페에서 작업하는 대학생 및 직장인")
        product_features = st.text_area(
            "핵심 소구점 (강조하고 싶은 특징)", 
            placeholder="예: 클릭 소음 90% 감소, 슬림한 디자인, 배터리 18개월 지속, 가벼운 무게",
            height=120
        )
        
    with col2:
        st.subheader("2. 📱 어디에 올릴 글인가요?")
        channel = st.selectbox(
            "마케팅 채널 선택", 
            ["인스타그램", "이메일 (뉴스레터)", "블로그", "페이스북", "링크드인", "트위터/X"]
        )
        tone_and_manner = st.selectbox(
            "톤앤매너 (글의 분위기)", 
            ["트렌디하고 톡톡 튀는 (MZ세대 타겟)", "전문적이고 신뢰감 있는", "감성적이고 따뜻한", "유머러스하고 재치 있는", "긴박감을 주는 (프로모션 마감 강조)"]
        )

st.markdown("---")

# --- 백엔드: 마케팅 글 생성 로직 ---
if st.button("✨ 마케팅 카피 생성하기", use_container_width=True):
    # 입력값 검증
    if not openai_api_key:
        st.error("⚠️ 서버 환경 변수에 OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
        st.stop()
        
    if not product_name or not product_features:
        st.warning("⚠️ 상품 이름과 핵심 소구점을 모두 입력해 주세요!")
        st.stop()

    with st.spinner(f"AI 카피라이터가 {channel}용 맞춤형 글을 작성하고 있습니다... ✍️"):
        try:
            # 1. LLM 설정
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)
            
            # 2. 프롬프트 엔지니어링 (채널별 가이드라인 주입)
            system_prompt = """당신은 10년 차 실력을 가진 탑 티어 마케팅 카피라이터입니다.
사용자가 제공하는 상품 정보, 타겟 고객, 채널 특성, 톤앤매너를 완벽하게 조합하여 전환율(CVR)을 극대화하는 마케팅 카피를 작성하세요.

[채널별 작성 가이드라인]
- 인스타그램: 시각적 상상을 자극하는 짧고 강렬한 첫 줄. 적절한 이모지 적극 활용. 문단은 보기 좋게 나누고, 마지막엔 핵심 해시태그 5~7개 배치.
- 이메일 (뉴스레터): 클릭을 유발하는 호기심 가득한 [제목]을 먼저 작성. 본문은 가독성 좋게 인사말-문제공감-해결책(상품)-행동촉구(CTA 버튼 문구) 순으로 구성.
- 블로그: 정보 전달과 스토리텔링 중심. 검색 엔진에 잘 걸리도록 자연스러운 키워드 배치. 서론-본론-결론의 짜임새 있는 구조.
- 페이스북: 공감대를 형성하는 질문으로 시작. 친근하게 소통하는 느낌을 주며, 댓글이나 공유를 유도하는 멘트 포함.
- 링크드인: B2B 및 전문가 타겟을 고려하여 논리적이고 통찰력 있는 비즈니스 톤 사용. 전문 용어를 적절히 섞어 신뢰감 부여.
- 트위터/X: 280자 이내로 매우 짧고 임팩트 있게 작성. 트렌디한 밈이나 위트 있는 펀치라인 활용.

요청받은 [톤앤매너]를 문장과 어투에 철저하게 반영하세요.
"""
            human_prompt = """
[작성 정보]
- 상품/서비스명: {product_name}
- 타겟 고객: {target_audience}
- 핵심 소구점: {product_features}
- 타겟 채널: {channel}
- 톤앤매너: {tone_and_manner}

위 정보를 바탕으로 완벽한 마케팅 카피를 작성해 주세요.
"""
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", human_prompt)
            ])
            
            # 3. 체인 생성 및 실행 (LCEL)
            chain = prompt_template | llm | StrOutputParser()
            
            result = chain.invoke({
                "product_name": product_name,
                "target_audience": target_audience if target_audience else "일반 대중",
                "product_features": product_features,
                "channel": channel,
                "tone_and_manner": tone_and_manner
            })
            
            # 4. 결과 출력
            st.success("🎉 작성이 완료되었습니다!")
            
            st.markdown(f"### 📋 {channel} 최적화 카피")
            st.info(result)
            
            # 복사하기 편하도록 코드 블록 형태로도 제공
            with st.expander("원문 복사하기"):
                st.code(result, language="markdown")
                
        except Exception as e:
            st.error(f"작성 중 오류가 발생했습니다: {str(e)}")
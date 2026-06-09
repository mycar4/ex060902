import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

# --- LangChain 공통 패키지 ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- 유튜브 패키지 ---
from langchain_community.document_loaders import YoutubeLoader

# --- 문서 요정 패키지 ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# 🚨 Streamlit의 페이지 기본 설정은 반드시 코드 최상단에 딱 한 번만 와야 합니다.
st.set_page_config(page_title="AI 업무 비서 대시보드", page_icon="🤖", layout="wide")

# 환경 변수 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ==========================================
# 1. 유튜브 요약기 기능
# ==========================================
def render_youtube_app():
    st.title("🎥 유튜브 영상 요약 및 번역기")
    st.markdown("유튜브 URL을 입력하면, 영상의 자막을 추출하여 핵심 내용을 요약하고 한국어로 번역해 줍니다.")

    youtube_url = st.text_input("유튜브 영상 링크를 입력하세요:", placeholder="https://www.youtube.com/watch?v=...")

    if st.button("요약 및 번역 시작", type="primary"):
        if not openai_api_key:
            st.error("⚠️ 서버 환경 변수에 OpenAI API Key가 설정되지 않았습니다.")
            st.stop()
        if not youtube_url:
            st.warning("⚠️ 유튜브 링크를 입력해 주세요.")
            st.stop()

        with st.spinner("자막을 추출하고 요약하는 중입니다... ⏳"):
            try:
                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False, language=["en", "ko", "ja", "es"])
                docs = loader.load()
                
                if not docs:
                    st.error("자막을 찾을 수 없는 영상입니다.")
                    st.stop()
                    
                transcript = docs[0].page_content
                video_title = docs[0].metadata.get("title", "제목 없음")
                
                llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
                prompt = PromptTemplate.from_template(
                    "당신은 요약 전문가입니다. 다음 유튜브 자막을 읽고:\n1. 영상의 핵심 주제 (1줄)\n2. 상세 요약 (3~5개 글머리 기호)\n양식에 맞춰 한국어로 작성해 주세요.\n\n자막: {transcript}"
                )
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({"transcript": transcript})
                
                st.success("✅ 요약이 완료되었습니다!")
                st.subheader(f"📺 영상 제목: {video_title}")
                st.markdown("---")
                st.markdown(result)
                
                with st.expander("원본 자막 내용 보기"):
                    st.text(transcript)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# ==========================================
# 2. 문서 요정 (RAG 챗봇) 기능
# ==========================================
def render_doc_fairy_app():
    st.title("🧚 나만의 맞춤형 문서 요정")
    st.markdown("문서를 업로드하면 AI 요정이 **내용을 요약해 드린 후**, 궁금한 점에 답변해 드립니다.")

    uploaded_file = st.file_uploader("분석할 파일을 선택하세요 (PDF, TXT 지원)", type=["pdf", "txt"])
    st.info("💡 파일을 올리면 즉시 요약 마법을 시작합니다!")

    if uploaded_file is not None:
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner(f"⏳ '{uploaded_file.name}' 분석 및 요약 중..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    if uploaded_file.name.endswith(".pdf"):
                        loader = PyPDFLoader(tmp_file_path)
                        docs = loader.load()
                    else:
                        loader = TextLoader(tmp_file_path, encoding="utf-8")
                        docs = loader.load()
                    os.unlink(tmp_file_path)
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
                    splits = text_splitter.split_documents(docs)
                    
                    if not splits:
                        st.error("❌ 텍스트를 추출할 수 없는 문서입니다.")
                        st.stop()
                    
                    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                    
                    llm_summary = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
                    full_text = "\n".join([doc.page_content for doc in docs])
                    summary_target_text = full_text[:15000] 
                    
                    summary_prompt = PromptTemplate.from_template("가장 중요한 핵심 내용을 3~5개의 글머리 기호로 요약해 주세요.\n\n[문서 내용]\n{text}")
                    summary_chain = summary_prompt | llm_summary | StrOutputParser()
                    summary_text = summary_chain.invoke({"text": summary_target_text})
                    
                    st.session_state.processed_file = uploaded_file.name
                    st.session_state.chat_history = [{"role": "assistant", "content": f"✨ **문서 학습 완료! 요약 내용은 다음과 같아요.**\n\n{summary_text}"}]
                    st.success(f"✅ {uploaded_file.name} 학습 완료!")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")
                    st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("문서에 대해 질문해 주세요!"):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        if "retriever" not in st.session_state:
            with st.chat_message("assistant"):
                st.warning("먼저 위에서 문서를 업로드해 주세요!")
        else:
            with st.chat_message("assistant"):
                with st.spinner("답변을 찾는 중..."):
                    retriever = st.session_state.retriever
                    retrieved_docs = retriever.invoke(user_query)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    langchain_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]) for msg in st.session_state.chat_history[-6:]]
                    
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "제공된 문서 내용(Context)을 바탕으로 한국어로 친절하게 답하세요.\n\nContext:\n{context}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])
                    chain = qa_prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": context, "chat_history": langchain_history, "input": user_query})
                    
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "user", "content": user_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

# ==========================================
# 3. 마케팅 카피라이터 기능
# ==========================================
def render_marketing_app():
    st.title("✍️ 맞춤형 마케팅 카피라이터")
    st.markdown("상품 정보와 타겟 고객을 입력하면, 채널 특성에 최적화된 마케팅 문구를 작성해 줍니다.")

    col1, col2 = st.columns(2)
    with col1:
        product_name = st.text_input("1. 상품/서비스명", placeholder="예: 무소음 무선 마우스 로지텍 M350")
        target_audience = st.text_input("2. 타겟 고객", placeholder="예: 카페에서 작업하는 대학생")
        product_features = st.text_area("3. 핵심 소구점", placeholder="예: 클릭 소음 90% 감소, 슬림한 디자인", height=100)
    with col2:
        channel = st.selectbox("4. 마케팅 채널", ["인스타그램", "이메일 (뉴스레터)", "블로그", "페이스북", "링크드인", "트위터/X"])
        tone_and_manner = st.selectbox("5. 톤앤매너", ["트렌디하고 톡톡 튀는", "전문적이고 신뢰감 있는", "감성적이고 따뜻한", "유머러스한", "긴박감을 주는 (프로모션)"])

    st.markdown("---")
    if st.button("✨ 마케팅 카피 생성하기", type="primary", use_container_width=True):
        if not openai_api_key:
            st.error("⚠️ 서버 환경 변수에 OpenAI API Key가 설정되지 않았습니다.")
            st.stop()
        if not product_name or not product_features:
            st.warning("⚠️ 상품 이름과 핵심 소구점을 입력해 주세요!")
            st.stop()

        with st.spinner(f"AI 카피라이터가 {channel}용 글을 작성하고 있습니다... ✍️"):
            try:
                llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=openai_api_key)
                system_prompt = """당신은 10년 차 마케팅 카피라이터입니다. 
채널 특성(인스타그램-해시태그/이모지, 이메일-제목/CTA, 블로그-정보/키워드, 등)에 맞춰 작성하세요.
톤앤매너를 철저히 반영하세요."""
                human_prompt = "[작성 정보]\n- 상품명: {product_name}\n- 타겟: {target_audience}\n- 소구점: {product_features}\n- 채널: {channel}\n- 톤앤매너: {tone_and_manner}\n\n위 정보로 카피를 작성해 주세요."
                
                prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
                chain = prompt_template | llm | StrOutputParser()
                result = chain.invoke({
                    "product_name": product_name, "target_audience": target_audience, 
                    "product_features": product_features, "channel": channel, "tone_and_manner": tone_and_manner
                })
                
                st.success("🎉 작성이 완료되었습니다!")
                st.markdown(f"### 📋 {channel} 최적화 카피")
                st.info(result)
                with st.expander("원문 복사하기"):
                    st.code(result, language="markdown")
            except Exception as e:
                st.error(f"작성 중 오류가 발생했습니다: {str(e)}")


# ==========================================
# 🚀 메인 화면 컨트롤러 (사이드바 라우팅)
# ==========================================
with st.sidebar:
    st.header("🤖 AI 업무 비서")
    st.markdown("원하는 기능을 선택하세요:")
    
    # 사이드바 라디오 버튼으로 메뉴 선택
    menu = st.radio(
        "메뉴",
        ["🏠 홈", "🎥 유튜브 영상 요약기", "🧚 문서 요정 (RAG)", "✍️ 마케팅 카피라이터"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Powered by LangChain & Streamlit")

# 선택된 메뉴에 따라 알맞은 함수를 실행하여 화면 렌더링
if menu == "🏠 홈":
    st.title("환영합니다! 🚀")
    st.markdown("""
    ### 여러분만의 강력한 AI 종합 업무 비서 대시보드입니다.
    좌측 사이드바에서 원하는 도구를 선택해 업무 효율을 10배 이상 끌어올려 보세요!
    
    * **🎥 유튜브 영상 요약기:** 긴 영상의 엑기스만 빠르게 뽑아볼 때
    * **🧚 문서 요정 (RAG):** 두꺼운 PDF 논문이나 매뉴얼과 대화하고 싶을 때
    * **✍️ 마케팅 카피라이터:** SNS, 블로그 등에 올릴 맞춤형 홍보 글이 막막할 때
    """)
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1200&q=80", use_column_width=True)

elif menu == "🎥 유튜브 영상 요약기":
    render_youtube_app()

elif menu == "🧚 문서 요정 (RAG)":
    render_doc_fairy_app()

elif menu == "✍️ 마케팅 카피라이터":
    render_marketing_app()
import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# .env 파일에서 환경 변수(OpenAI API 키) 로드
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- UI 설정 ---
st.set_page_config(page_title="맞춤형 문서 요정", page_icon="🧚", layout="wide")
st.title("🧚 나만의 맞춤형 문서 요정 (RAG Chatbot)")
st.markdown("문서를 업로드하면 AI 요정이 **내용을 요약해 드린 후**, 궁금한 점에 답변해 드립니다.")

# --- 사이드바: 파일 업로더 ---
with st.sidebar:
    st.header("📂 문서 업로드")
    uploaded_file = st.file_uploader(
        "분석할 파일을 선택하세요 (PDF, TXT 지원)", 
        type=["pdf", "txt"]
    )
    st.markdown("---")
    st.info("💡 파일을 올리면 즉시 요약 마법을 시작합니다!")

# --- 백엔드: 문서 처리 및 자동 요약 로직 ---
if uploaded_file is not None:
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner(f"⏳ '{uploaded_file.name}' 분석 및 요약 중..."):
            try:
                # 1. 파일 임시 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # 2. 로더 선택
                if uploaded_file.name.endswith(".pdf"):
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                elif uploaded_file.name.endswith(".txt"):
                    loader = TextLoader(tmp_file_path, encoding="utf-8")
                    docs = loader.load()
                
                os.unlink(tmp_file_path) # 임시파일 삭제
                
                # 3. 텍스트 청크 분할 (RAG용)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
                splits = text_splitter.split_documents(docs)
                
                if not splits:
                    st.sidebar.error("❌ 텍스트를 추출할 수 없는 문서입니다.")
                    st.stop()
                
                # 4. 벡터 저장소 구축
                embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
                vectorstore = FAISS.from_documents(splits, embeddings)
                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # 5. [신규] 에러 안 나는 가벼운 요약 방식 적용 (LCEL)
                llm_summary = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)
                
                # 문서의 전체 내용을 하나의 텍스트로 합칩니다 (너무 길면 앞부분 15,000자만 요약에 사용해 속도 확보)
                full_text = "\n".join([doc.page_content for doc in docs])
                summary_target_text = full_text[:15000] 
                
                summary_prompt = PromptTemplate.from_template(
                    "당신은 전문 요약가입니다. 다음 문서 내용을 읽고 가장 중요한 핵심 내용을 3~5개의 글머리 기호로 요약해 주세요.\n\n[문서 내용]\n{text}"
                )
                
                summary_chain = summary_prompt | llm_summary | StrOutputParser()
                summary_text = summary_chain.invoke({"text": summary_target_text})
                
                # 6. 세션 상태 초기화 및 요약본을 첫 대화로 삽입
                st.session_state.processed_file = uploaded_file.name
                st.session_state.chat_history = [
                    {"role": "assistant", "content": f"✨ **문서 학습을 완료했습니다! 요약 내용은 다음과 같아요.**\n\n{summary_text}"}
                ]
                st.sidebar.success(f"✅ {uploaded_file.name} 학습 완료!")
                
            except Exception as e:
                st.sidebar.error(f"오류 발생: {str(e)}")
                st.stop()

# --- 메인 화면: 챗봇 대화 인터페이스 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 대화 기록 렌더링
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if user_query := st.chat_input("요약 내용을 확인하고 질문해 주세요!"):
    with st.chat_message("user"):
        st.markdown(user_query)
    
    if "retriever" not in st.session_state:
        with st.chat_message("assistant"):
            st.warning("먼저 왼쪽에서 문서를 업로드해 주세요!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("답변을 찾는 중..."):
                try:
                    retriever = st.session_state.retriever
                    retrieved_docs = retriever.invoke(user_query)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    langchain_history = [
                        HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                        for msg in st.session_state.chat_history[-6:]
                    ]
                    
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
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")
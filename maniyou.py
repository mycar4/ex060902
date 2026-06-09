from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import YoutubeLoader

from langchain_openai import ChatOpenAI
chat_model = ChatOpenAI()


# 유튜브 URL로부터 자막을 로드하기 위한 로더를 생성합니다.
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=h3EqCo2HxUA",
    add_video_info=True,
    language=["ko", "en"] # 한국어 자막을 우선, 없으면 영어 자막을 가져옵니다.
)

# 문서를 로드합니다.
docs = loader.load()

# 로드된 문서의 내용을 하나의 문자열로 합칩니다.
content = "".join([doc.page_content for doc in docs])

# 모델에 설명을 요청하고 결과를 출력합니다.
response = chat_model.invoke(content + "\n\n 위 내용에 대해 설명해줘")
print(response.content)
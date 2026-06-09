import os
import urllib.error
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 환경변수 로드 (.env 파일에 OPENAI_API_KEY가 있어야 합니다)
load_dotenv()

def main():
    print("="*60)
    print(" 📺 유튜브 영상 요약기 (터미널 버전)")
    print("="*60)

    # 터미널에서 직접 사용자에게 URL을 입력받습니다.
    url = input("\n🔗 유튜브 링크를 입력하세요 (종료하려면 엔터): ")
    
    if not url.strip():
        print("프로그램을 종료합니다.")
        return

    print("\n⏳ 자막을 추출하는 중입니다...")

    try:
        # 1. 유튜브 자막 및 메타데이터 로드
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["ko", "en"] # 한국어 우선, 없으면 영어
        )
        docs = loader.load()
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print("\n❌ 유튜브 영상 정보를 가져오는데 실패했습니다 (HTTP 400: Bad Request).")
            print("   'pytube' 라이브러리 버전이 오래되어 발생하는 흔한 문제입니다.")
            print("   아래 명령어를 터미널에 실행하여 라이브러리를 업데이트해 보세요.")
            print("\n   pip install --upgrade pytube\n")
        else:
            print(f"\n❌ 영상 정보나 자막을 가져오는데 실패했습니다 (HTTP {e.code}):\n{e}")
        return
    except Exception as e:
        print(f"\n❌ 영상 정보나 자막을 가져오는 중 예상치 못한 오류가 발생했습니다:\n{e}")
        return

    if not docs:
        print("\n❌ 추출할 수 있는 자막이 없습니다.")
        return

    # 자막 내용과 영상 정보 추출
    content = docs[0].page_content
    title = docs[0].metadata.get('title', '제목 없음')
    author = docs[0].metadata.get('author', '알 수 없는 채널')

    print(f"✅ 영상 인식 완료: [{title}] (채널: {author})")
    print("🤖 AI가 내용을 분석하고 요약 중입니다. 잠시만 기다려주세요...\n")

    # 2. LLM 모델 설정
    chat_model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

    # 3. 요약 및 번역을 위한 프롬프트 템플릿 설정
    prompt = PromptTemplate.from_template(
        """
        당신은 뛰어난 전문 요약가이자 번역가입니다. 
        다음은 유튜브 영상의 자막입니다. 이 내용을 읽고 한국어로 알기 쉽게 요약 및 번역해 주세요.

        [영상 정보]
        - 제목: {title}
        - 채널명: {author}

        [영상 자막 원본]
        {text}

        [출력 형식 가이드]
        1. 📝 세 줄 요약: 영상의 가장 핵심적인 내용을 3줄로 명확하게 요약하세요.
        2. 💡 주요 내용 정리: 상세한 주요 포인트를 불릿 포인트(-)를 사용하여 정리하세요. 전문 용어가 있다면 쉽게 풀어서 설명하세요.
        3. 🎯 결론 및 시사점: 이 영상이 전달하고자 하는 최종 메시지를 작성하세요.
        """
    )

    # 4. 체인 구성 및 실행
    chain = prompt | chat_model
    
    try:
        response = chain.invoke({
            "title": title,
            "author": author,
            "text": content
        })

        # 5. 결과 출력
        print("\n" + "="*20 + " 📊 요약 결과 " + "="*20 + "\n")
        print(response.content)
        print("\n" + "="*54)
        
    except Exception as e:
        print(f"\n❌ AI 요약 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
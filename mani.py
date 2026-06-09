from langchain_community.document_loaders import PyPDFLoader




# file_path = "unsu.pdf"
# loader = PyPDFLoader(file_path)

# PDF 로더 인스턴스 생성
loader = PyPDFLoader("unsu.pdf")
# PDF 파일에서 페이지를 로드하고 분할하여 페이지 객체 리스트로 반환
pages = loader.load_and_split()

# 데이터 확인 및 출력 
if len(pages) > 1:
    print("---[두 번째 페이지 객체 전체 출력]---")
    print(pages[1])

    print("---[두 번째 페이지 객체 전체 출력]---")
    print(pages[1].page_content)

else:
    print(f"PDF 파일에서 페이지가 {len(pages)}개 있습니다.")


# print(pages[3])
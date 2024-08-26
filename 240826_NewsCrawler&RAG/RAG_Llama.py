import os
import shutil
from datetime import datetime
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# 1. Ollama를 사용하여 로컬에서 Llama 모델 불러오기
llm = Ollama(model="llama3.1")  # Ollama 설치 및 llama3.1 모델 준비 필요

# 2. 크롤링된 CSV 파일 경로 설정
CSV_FILE_PATH = "./crawling_results/240826_AI_news_crawling.csv"

# 3. CSV 파일 로드 (메타데이터 처리 개선)
loader = CSVLoader(
    file_path=CSV_FILE_PATH,
    encoding="utf-8-sig",
    csv_args={'delimiter': ','},
    source_column="link"
)
raw_documents = loader.load()

# 날짜 처리 함수
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now()

# 메타데이터 정리 및 문서 생성
documents = [
    Document(
        page_content=doc.page_content,
        metadata={
            "source": doc.metadata.get("source", "Unknown"),
            "date": parse_date(doc.metadata.get("date", "")).strftime("%Y-%m-%d %H:%M:%S"),
            "title": doc.metadata.get("title", "Untitled")
        }
    ) for doc in raw_documents
]

# 4. 문서 로드 및 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# 5. 임베딩 설정
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 6. 커스텀 메타데이터 필터링 함수
def custom_filter_metadata(doc: Document) -> Document:
    filtered_metadata = {}
    for key, value in doc.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            filtered_metadata[key] = value
    return Document(page_content=doc.page_content, metadata=filtered_metadata)

# 7. Chroma를 사용하여 벡터 스토어 생성 (메타데이터 필터링 추가)
filtered_texts = [custom_filter_metadata(text) for text in texts]
vector_store = Chroma.from_documents(filtered_texts, embeddings, persist_directory="./chroma_db")

# 8. 프롬프트 템플릿 설정 (새로운 버전 유지)
prompt_template = PromptTemplate(
    template="""
    당신은 생성형 AI 기술에 대한 깊은 이해와 넓은 시야를 가진 테크 에반젤리스트입니다. 
    최신 AI 기술 동향에 대한 종합적이고 통찰력 있는 블로그 포스트를 작성해주세요.

    다음 단계를 따라 포스트를 작성하세요:

    1. 정보 분석 및 선별:
       - 제공된 모든 뉴스와 정보를 꼼꼼히 검토하세요.
       - AI 기술 발전의 큰 그림을 그릴 수 있는 중요한 내용을 선별하세요.
       - 각 소식의 기술적, 사회적, 경제적 영향을 고려하세요.

    2. 포스트 구조:
       a. 도입부 (약 500자):
          - 현재 AI 기술 동향의 큰 그림을 제시합니다.
          - 독자의 관심을 끌 수 있는 흥미로운 통계나 사례로 시작하세요.

       b. 주요 트렌드 분석 (각 800-1000자, 3-4개 주제):
          - 가장 중요하고 영향력 있는 AI 트렌드를 심도 있게 다룹니다.
          - 각 트렌드에 대해:
            * 기술적 의의와 혁신성을 설명합니다.
            * 실제 적용 사례나 잠재적 응용 분야를 제시합니다.
            * 관련 기업, 연구소, 전문가의 견해를 인용합니다.
            * 해당 기술의 미래 전망과 잠재적 영향을 분석합니다.

       c. 기술 간 시너지 및 통합 동향 (약 800자):
          - 다양한 AI 기술들이 어떻게 융합되고 있는지 설명합니다.
          - 이러한 통합이 가져올 수 있는 혁신적 변화를 예측합니다.

       d. 도전 과제 및 윤리적 고려사항 (약 600자):
          - AI 기술 발전에 따른 주요 도전 과제를 논의합니다.
          - 윤리적, 사회적 영향에 대한 균형 잡힌 시각을 제시합니다.

       e. 미래 전망 및 결론 (약 700자):
          - 분석한 트렌드를 바탕으로 AI 기술의 미래를 전망합니다.
          - 기업, 개발자, 일반 사용자들에게 주는 시사점을 제시합니다.

    3. 작성 지침:
       - 테크 에반젤리스트로서의 전문성과 통찰력을 보여주는 톤으로 작성하세요.
       - 기술적 정확성과 일반 독자의 이해도 사이의 균형을 유지하세요.
       - 구체적인 사례, 통계, 전문가 인용을 활용하여 내용의 신뢰성을 높이세요.
       - 각 섹션에 적절한 소제목을 사용하여 내용을 체계적으로 구성하세요.
       - 각 주요 정보나 주장에 대해 출처를 명시하세요. 형식: [출처: 기사 제목, 게시일]

    다음은 제공된 컨텍스트 정보입니다. 이 정보를 바탕으로 위의 구조와 지침에 따라 포스트를 작성해주세요:

    {context}

    마지막으로, 포스트의 끝에 모든 참고 문헌을 다음 형식으로 정리해주세요:

    참고 문헌:
    1. [기사 제목] - [출처], [게시일]
    2. [기사 제목] - [출처], [게시일]
    ...

    이 참고 문헌 목록에는 포스트 내에서 언급된 모든 출처가 포함되어야 합니다.
    """,
    input_variables=["context"]
)

# 9. RetrievalQA 체인 설정
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# 메타데이터 처리 및 오류 핸들링 함수
def safe_get_metadata(doc, key, default="N/A"):
    try:
        return doc.metadata.get(key, default)
    except AttributeError:
        return default

# 10. 블로그 포스트 저장 함수
def save_blog_post(content):
    save_dir = "./blog_posts"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    today = datetime.now().strftime("%Y%m%d")
    file_name = f"{today}.txt"
    file_path = os.path.join(save_dir, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return file_path

# 11. RAG를 통한 질문/응답 함수 정의
def generate_blog_post():
    query = "최근 생성형 AI 기술 동향과 주요 모델들의 발전 사항"
    response = rag_chain({"query": query})
    blog_post = response["result"]
    file_path = save_blog_post(blog_post)
    return blog_post, response["source_documents"], file_path

# 12. 자동으로 블로그 포스트 생성
blog_post, sources, file_path = generate_blog_post()
print("\n생성된 블로그 포스트:")
print(blog_post)
print(f"\n블로그 포스트가 다음 위치에 저장되었습니다: {file_path}")
print("\n참고한 문서:")
for i, doc in enumerate(sources):
    date = safe_get_metadata(doc, 'date')
    title = safe_get_metadata(doc, 'title')
    source = safe_get_metadata(doc, 'source')
    content = doc.page_content[:100] if hasattr(doc, 'page_content') else "내용 없음"
    
    print(f"{i+1}. 날짜: {date}, 제목: {title}")
    print(f"   출처: {source}")
    print(f"   내용: {content}...")

# 디버깅을 위한 메타데이터 출력
print("\n\n--- 디버깅: 문서 메타데이터 ---")
for i, doc in enumerate(sources):
    print(f"문서 {i+1} 메타데이터:")
    print(doc.metadata if hasattr(doc, 'metadata') else "메타데이터 없음")
    print("---")
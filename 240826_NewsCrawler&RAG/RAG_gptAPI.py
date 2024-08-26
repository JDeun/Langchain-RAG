import os
import shutil
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 실제 사용 시 이 부분을 여러분의 API 키로 교체하세요

# 1. gpt-4o-mini 모델 불러오기
llm = ChatOpenAI(model_name="gpt-4o-mini")

# 2. 크롤링된 CSV 파일 경로 설정
CSV_FILE_PATH = "./crawling_results/240826_AI_news_crawling.csv"

# 3. CSV 파일 로드
loader = CSVLoader(
    file_path=CSV_FILE_PATH,
    encoding="utf-8-sig",
    csv_args={'delimiter': ','},
    source_column="link"
)
raw_documents = loader.load()

# 4. 문서 로드 및 분할
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(raw_documents)

# 5. 임베딩 설정
embeddings = OpenAIEmbeddings()

# 6. Chroma 데이터베이스 초기화
CHROMA_DB_PATH = "./chroma_db"
if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)  # 기존 데이터베이스 삭제

# 7. Chroma를 사용하여 벡터 스토어 생성
vector_store = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_PATH)

# 8. 프롬프트 템플릿 설정
prompt_template = PromptTemplate(
    template="""
    당신은 AI 기술에 대해 깊이 이해하고 있으며, 사람들에게 AI 기술 동향을 쉽게 설명하는 테크 에반젤리스트입니다. 
    최신 AI 기술 동향에 대한 종합적이고 통찰력 있는 블로그 포스트를 작성해주세요. 독자가 쉽게 이해할 수 있도록 친근하고 대화체의 문체로 작성하세요.

    다음 단계를 따라 포스트를 작성하세요:

    1. 정보 분석 및 선별:
       - 제공된 모든 뉴스와 정보를 신중하게 검토하세요.
       - AI 기술 발전의 큰 그림을 이해할 수 있는 중요한 내용을 선택하세요.
       - 각 소식이 기술, 사회, 경제에 미치는 영향을 고려하세요.
       - 최신성을 유지하기 위해 코드 실행일 기준 7일 이내의 기사만 사용하세요.

    2. 포스트 구조:
       a. 도입부 (약 1000자):
          - 현재 AI 기술의 전반적인 동향을 쉽게 풀어서 소개합니다.
          - 흥미로운 통계나 실생활 사례로 독자의 관심을 끌어보세요.
          - 변화가 우리 삶에 미칠 영향을 강조하며, 독자에게 주제를 쉽게 이해시키세요.

       b. 주요 트렌드 분석 (각 주제별 2000-3000자, 3-4개 주제):
          - 가장 주목할 만한 AI 트렌드 몇 가지를 깊이 있게 다룹니다.
          - 각 트렌드에 대해:
            * 해당 기술의 중요성과 혁신성을 설명하세요.
            * 실제 적용 사례나 앞으로의 가능성을 이야기하세요.
            * 관련된 기업, 연구기관, 또는 전문가의 견해를 인용하세요.
            * 기술이 미래에 미칠 영향과 전망을 분석하세요.
            * 각 주제를 2-3개의 문단으로 나누어 부드럽게 연결하며, 흐름이 자연스럽도록 작성하세요.
            * 주요 트렌드에서 다룬 내용은 단신 뉴스에 중복되지 않도록 주의하세요.

       c. 단신 뉴스 (약 800자):
          - 주요 트렌드 분석에서 다루지 않은 중요한 AI 관련 소식을 짧게 요약합니다.
          - 각 소식을 2-3 문장으로 간결하게 정리하세요.
          - 독자가 쉽게 이해할 수 있도록 일상적인 언어로 설명하세요.

    3. 작성 지침:
       - 친근하고 이해하기 쉬운 톤으로, 마치 대화를 나누는 것처럼 작성하세요.
       - 복잡한 기술적 용어는 간단히 풀어 설명하고, 일반 독자도 이해할 수 있는 수준을 유지하세요.
       - 실제 사례, 통계, 전문가 의견을 인용하여 신뢰성을 높이세요.
       - 줄글 형식으로 내용을 구성하여 독자가 자연스럽게 글의 흐름을 따라갈 수 있게 하세요.
       - 주요 정보나 주장에 대해 출처를 명시하세요. 형식: [출처: 기사 제목, 게시일]

    다음은 제공된 컨텍스트 정보입니다. 이 정보를 바탕으로 위의 구조와 지침에 따라 포스트를 작성해주세요:

    {context}

    마지막으로, 포스트의 끝에 모든 참고 문헌을 다음 형식으로 정리해주세요:

    참고 문헌:
    1. [기사 제목] - [게시일], [URL]
    2. [기사 제목] - [게시일], [URL]
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

# 11. 기존 블로그 포스트 로드
def load_existing_posts(directory):
    posts = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path)
            posts.extend(loader.load())
    return posts

# 12. 유사도 계산
def calculate_similarity(new_post, existing_posts, embeddings):
    new_embedding = embeddings.embed_documents([new_post])[0]
    existing_embeddings = embeddings.embed_documents([post.page_content for post in existing_posts])
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    return similarities

# 13. 유사도 체크 및 재생성
def check_similarity_and_regenerate(new_post, existing_posts, embeddings, similarity_threshold=0.8):
    similarities = calculate_similarity(new_post, existing_posts, embeddings)
    if np.max(similarities) > similarity_threshold:
        print("유사한 내용의 글이 이미 존재합니다. 새로운 주제로 글을 재생성합니다.")
        return True
    return False

# 14. RAG를 통한 질문/응답 함수 정의 (최대 5회 시도)
def generate_blog_post():
    query = "최근 생성형 AI 기술 동향과 주요 모델들의 발전 사항"
    existing_posts = load_existing_posts("./blog_posts")
    
    for attempt in range(5):
        response = rag_chain({"query": query})
        blog_post = response["result"]
        
        if not check_similarity_and_regenerate(blog_post, existing_posts, embeddings):
            file_path = save_blog_post(blog_post)
            return blog_post, response["source_documents"], file_path
        
        print(f"시도 {attempt + 1}/5: 유사한 내용 감지. 재생성 중...")
        query = f"최근 생성형 AI 기술 동향과 주요 모델들의 발전 사항 (이전과 다른 관점에서, 시도 {attempt + 2})"
    
    print("5회 시도 후에도 유니크한 내용을 생성하지 못했습니다. 프로그램을 종료합니다.")
    return None, None, None

# 메인 실행 부분
if __name__ == "__main__":
    blog_post, sources, file_path = generate_blog_post()
    if blog_post:
        print("\n생성된 블로그 포스트:")
        print(blog_post)
        print(f"\n블로그 포스트가 다음 위치에 저장되었습니다: {file_path}")
        print("\n참고한 문서:")
        for i, doc in enumerate(sources):
            print(f"{i+1}. 제목: {doc.metadata.get('title', 'N/A')}")
            print(f"   출처: {doc.metadata.get('source', 'N/A')}")
            print(f"   내용: {doc.page_content[:100]}...")
    else:
        print("블로그 포스트 생성에 실패했습니다.")
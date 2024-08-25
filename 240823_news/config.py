from datetime import datetime, timedelta

# 검색 쿼리 설정
search_queries = ["ChatGPT", "Gemini", "Claude", "Grok", "Llama", "AI", "머신러닝", "딥러닝", "자연어처리", "컴퓨터 비전"]

# 동적으로 주제 생성
def generate_newsletter_topic(search_queries):
    return "주요 AI 모델 및 기술의 최신 동향과 산업적 영향: 주간 AI 트렌드 리포트"

newsletter_topic = generate_newsletter_topic(search_queries)

config = {
    'SEARCH_QUERY': search_queries,
    'NEWSLETTER_TOPIC': newsletter_topic,
    'TARGET_WORD_COUNT': 1500,  # 뉴스레터 목표 단어 수
    'HOT_ISSUES_COUNT': 2,
    'TECH_SNAPSHOTS_COUNT': 2,
    'AI_SHORTS_COUNT': 5,
    'RECENT_NEWS_DAYS': 7,  # 최근 뉴스 기준 일수
}

# LLM 설정
LLAMA_MODEL_NAME = "llama3.1"
TEMPERATURE = 0.3
MAX_TOKENS = None
TOP_P = 0.95

# 저장 디렉토리 설정
CRAWLING_SAVE_DIR = './crawling_save'
CONTENT_SAVE_DIR = './contents_save'

# 임베딩 모델 설정
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG 시스템 설정
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_K = 5

# 뉴스 크롤링 설정
SEARCH_QUERY = search_queries
START_PAGE = 1
END_PAGE = 50

# 최근 뉴스 기준 날짜
RECENT_NEWS_DATE = datetime.now() - timedelta(days=config['RECENT_NEWS_DAYS'])
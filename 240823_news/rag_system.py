from datetime import datetime, timedelta
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from config import LLAMA_MODEL_NAME, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TEMPERATURE, MAX_TOKENS, TOP_P, SEARCH_K, config
import logging
from tqdm import tqdm
from utils import parse_date, is_recent_news

class RAGSystem:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = ChatOllama(
            model=LLAMA_MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P
        )
        self.config = config
        self.hot_issue_prompt = """
        '{query}'와 관련된 AI 기술의 최신 트렌드와 발전 사항을 분석해주세요.
        다음 사항을 포함해 주세요:
        1. 최근 1주일 내의 주요 기술적 돌파구
        2. 이 발전이 산업과 사회에 미치는 잠재적 영향
        3. 관련된 윤리적 고려사항이나 도전 과제
        4. 향후 1-3개월 동안의 발전 전망
        최소 3-4문장으로 요약해 주세요. 뉴스의 발행 날짜도 포함해 주세요.
        """
        self.tech_snapshot_prompt = """
        다음 키워드와 관련된 가장 중요한 최근 AI 기술 트렌드를 설명해주세요: {tech_snapshot_query}
        다음 정보를 포함해 주세요:
        1. 기술 이름
        2. 상세 설명 (최소 3-4문장)
        3. 실제 응용 사례 또는 잠재적 영향
        최근 1주일 내의 정보만 포함해 주세요. 뉴스의 발행 날짜도 포함해 주세요.
        """
        self.ai_shorts_prompt = """
        전 세계적으로 최근 1주일 내의 AI 발전에 대한 5개의 간단한 뉴스 업데이트를 제공해 주세요. 
        각 업데이트는 1-2문장 길이여야 합니다. 
        각 뉴스의 발행 날짜도 포함해 주세요.
        """

    def create_vectorstore(self, news_df):
        news_df['parsed_date'] = news_df['date'].apply(parse_date)
        valid_dates = news_df['parsed_date'].notnull()
        news_df = news_df[valid_dates]
        
        if news_df.empty:
            logging.error("유효한 날짜가 있는 뉴스가 없습니다.")
            return None
        
        one_week_ago = datetime.now() - timedelta(days=7)
        recent_news = news_df[news_df['parsed_date'] >= one_week_ago]
        
        logging.info(f"전체 뉴스 기사: {len(news_df)}")
        logging.info(f"최근 뉴스 기사 (1주일 이내): {len(recent_news)}")
        
        if recent_news.empty:
            logging.warning("최근 1주일 내 뉴스가 없습니다. 전체 뉴스 중 최신 20개를 사용합니다.")
            recent_news = news_df.sort_values('parsed_date', ascending=False).head(20)


        documents = [Document(page_content=row['content'],
                              metadata={"source": row['link'],
                                        "date": row['date'],
                                        "title": row['title'],
                                        "query": row['query']})
                     for _, row in recent_news.iterrows() if row['content'].strip()]

        if not documents:
            logging.error("유효한 문서가 없습니다.")
            return None

        texts = self.text_splitter.split_documents(documents)
        
        logging.info(f"텍스트 청크 수: {len(texts)}")
        
        if not texts:
            logging.error("분할 후 생성된 텍스트 청크가 없습니다.")
            return None

        logging.info(f"평균 청크 크기: {sum(len(t.page_content) for t in texts) / len(texts)}")

        try:
            return Chroma.from_documents(texts, self.embeddings)
        except Exception as e:
            logging.error(f"벡터 저장소 생성 중 오류 발생: {e}")
            return None

    def setup_retrieval_qa(self, vectorstore):
        retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})
        
        template = """다음의 문맥을 사용하여 질문에 답하세요. 
        답을 모르겠다면, 모른다고 말하고 답변을 만들어내려 하지 마세요. 
        최소 3-4문장으로 답변하고 상세하게 유지하세요.
        {context}
        질문: {question}
        답변:"""
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

    def get_relevant_news(self, qa_chain, queries):
        all_results = {
            "hot_issues": [],
            "tech_snapshots": [],
            "ai_shorts": None
        }
        
        for query in queries[:self.config['HOT_ISSUES_COUNT']]:
            hot_issue = qa_chain({"query": self.hot_issue_prompt.format(query=query)})
            if '최근 관련 뉴스 없음' not in hot_issue['result']:
                all_results["hot_issues"].append({"query": query, "result": hot_issue['result']})
            else:
                logging.info(f"'{query}'에 대한 최근 뉴스가 없습니다.")

        if not all_results["hot_issues"]:
            logging.warning("모든 쿼리에 대해 최근 관련 뉴스가 없습니다.")
            return None

        tech_snapshot_query = ", ".join([item["query"] for item in all_results["hot_issues"]])
        tech_snapshots = qa_chain({"query": self.tech_snapshot_prompt.format(tech_snapshot_query=tech_snapshot_query)})
        all_results["tech_snapshots"] = tech_snapshots['result']

        ai_shorts = qa_chain({"query": self.ai_shorts_prompt})
        all_results["ai_shorts"] = ai_shorts['result']

        return all_results
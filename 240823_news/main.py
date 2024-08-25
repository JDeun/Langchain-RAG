import pandas as pd
from news_crawler import NewsCrawler
from rag_system import RAGSystem
from newsletter_generator import NewsletterGenerator
from content_manager import ContentManager
from prompts import prompts
from config import config, SEARCH_QUERY, START_PAGE, END_PAGE
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    crawler = NewsCrawler(search_queries=SEARCH_QUERY)
    
    latest_csv = crawler.get_latest_csv()
    
    if not latest_csv:
        logging.info("오늘자 크롤링 데이터가 없습니다. 새로 크롤링을 시작합니다...")
        csv_path = crawler.crawl_news(start_pg=START_PAGE, end_pg=END_PAGE)
    else:
        csv_path = latest_csv
        logging.info(f"최신 크롤링 데이터를 사용합니다: {csv_path}")

    news_df = pd.read_csv(csv_path)
    logging.info(f"로드된 뉴스 개수: {len(news_df)}")

    rag_system = RAGSystem()
    vectorstore = rag_system.create_vectorstore(news_df)
    
    if vectorstore is None:
        logging.error("Vector store creation failed. Exiting...")
        return

    qa_chain = rag_system.setup_retrieval_qa(vectorstore)

    newsletter_generator = NewsletterGenerator(rag_system.llm, prompts, config)
    content_manager = ContentManager(search_queries=SEARCH_QUERY)

    context = rag_system.get_relevant_news(qa_chain, SEARCH_QUERY)
    
    newsletter_content = newsletter_generator.generate_content(context, SEARCH_QUERY)

    if content_manager.check_similarity(newsletter_content):
        logging.warning("생성된 콘텐츠가 기존 콘텐츠와 유사합니다. 다시 시도합니다...")
        newsletter_content = content_manager.get_unique_content(newsletter_content)
        if not newsletter_content:
            logging.error("유니크한 콘텐츠를 생성하지 못했습니다.")
            return

    current_date = datetime.now().strftime("%y%m%d")
    content_manager.save_content(newsletter_content, f"{current_date}_AI_newsletter.md")
    logging.info("뉴스레터 내용이 생성되어 저장되었습니다.")
    logging.info(f"생성된 뉴스레터 내용:\n{newsletter_content}")

if __name__ == "__main__":
    main()
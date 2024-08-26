import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import feedparser
from urllib.parse import quote, urljoin
import os
from datetime import datetime, timedelta

# 필요한 모듈 정의
CRAWLING_SAVE_DIR = "./crawling_results"  # 결과 저장 경로
RECENT_NEWS_DATE = datetime.now() - timedelta(days=7)  # 최근 7일 내의 뉴스만 수집

# 날짜 파싱 함수 정의
def parse_date(date_str):
    try:
        # data-date-time 형식으로 들어오는 경우를 처리
        return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        try:
            # 다른 일반적인 날짜 형식 처리
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

class NewsCrawler:
    def __init__(self, search_queries):
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/98.0.4758.102"}
        self.google_news_url = "https://news.google.com/rss/search?q={}&hl=ko&gl=KR&ceid=KR:ko"
        self.search_queries = search_queries

    def make_naver_url(self, search, start_pg, end_pg):
        urls = []
        for i in range(start_pg, end_pg + 1):
            page = (i - 1) * 10 + 1
            url = f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={quote(search)}&start={page}"
            urls.append(url)
        return urls

    def crawl_naver_news(self, search, start_pg, end_pg):
        urls = self.make_naver_url(search, start_pg, end_pg)
        news_urls = []

        for url in urls:
            try:
                original_html = requests.get(url, headers=self.headers)
                html = BeautifulSoup(original_html.text, "html.parser")
                url_naver = html.select("div.info_group > a.info")
                for i in url_naver:
                    if "news.naver.com" in i.get('href', ''):
                        news_urls.append(i['href'])
                    elif i.get('class') == ['info']:
                        news_urls.append(urljoin("https://search.naver.com", i['href']))
            except requests.RequestException as e:
                print(f"Error requesting URL {url}: {e}")

        return list(set(news_urls))

    def crawl_google_news(self, search):
        encoded_search = quote(search)
        feed_url = self.google_news_url.format(encoded_search)
        feed = feedparser.parse(feed_url)
        return [entry.link for entry in feed.entries]

    def get_news_content(self, url):
        try:
            news = requests.get(url, headers=self.headers)
            news_html = BeautifulSoup(news.text, "html.parser")

            title = news_html.select_one("h2#title_area") or news_html.select_one("h3#articleTitle") or news_html.select_one("h3.tts_head") or news_html.select_one("h2.end_tit")
            
            content = news_html.select_one("#dic_area") or news_html.select_one("#articleBodyContents") or news_html.select_one("div#article_body") or news_html.select_one("div#newsEndContents")
            
            # 여기서 날짜 정보를 가져옴
            date_element = news_html.select_one("span.media_end_head_info_datestamp_time")
            date = date_element['data-date-time'] if date_element and 'data-date-time' in date_element.attrs else None

            if title and content and date:
                title = title.get_text(strip=True)
                content = content.get_text(strip=True)
                return title, content, date
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
        
        return None, None, None

    def get_latest_csv(self):
        csv_files = [f for f in os.listdir(CRAWLING_SAVE_DIR) if f.endswith('_AI_news_crawling.csv')]
        if not csv_files:
            return None
        latest_file = max(csv_files, key=lambda x: os.path.getctime(os.path.join(CRAWLING_SAVE_DIR, x)))
        latest_file_path = os.path.join(CRAWLING_SAVE_DIR, latest_file)
        
        file_date = datetime.fromtimestamp(os.path.getctime(latest_file_path)).date()
        today = datetime.now().date()
        
        if file_date == today:
            return latest_file_path
        return None

    def crawl_news(self, start_pg, end_pg):
        today_csv = self.get_latest_csv()
        if today_csv:
            print(f"오늘 날짜의 크롤링 데이터가 이미 존재합니다: {today_csv}")
            return today_csv

        all_news_data = []

        for query in self.search_queries:
            print(f"'{query}' 검색어에 대한 뉴스 크롤링 시작...")
            naver_urls = self.crawl_naver_news(query, start_pg, end_pg)
            google_urls = self.crawl_google_news(query)

            all_urls = list(set(naver_urls + google_urls))

            for url in tqdm(all_urls, desc=f"{query} 뉴스 크롤링 진행 중"):
                title, content, date = self.get_news_content(url)
                parsed_date = parse_date(date) if date else None
                if title and content:
                    if parsed_date and parsed_date >= RECENT_NEWS_DATE:
                        all_news_data.append({
                            'query': query,
                            'date': date,
                            'parsed_date': parsed_date,
                            'title': title,
                            'link': url,
                            'content': content
                        })
                    elif not parsed_date:
                        print(f"날짜 정보를 파싱할 수 없습니다. 최신 뉴스로 가정합니다: {url}")
                        all_news_data.append({
                            'query': query,
                            'date': 'Unknown',
                            'parsed_date': datetime.now(),  # 현재 시간으로 설정
                            'title': title,
                            'link': url,
                            'content': content
                        })

        df = pd.DataFrame(all_news_data)
        
        # 날짜순으로 정렬
        df = df.sort_values('parsed_date', ascending=False)

        os.makedirs(CRAWLING_SAVE_DIR, exist_ok=True)
        current_date = datetime.now().strftime("%y%m%d")
        filename = f"{current_date}_AI_news_crawling.csv"
        filepath = os.path.join(CRAWLING_SAVE_DIR, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"크롤링 결과가 {filepath}에 저장되었습니다.")

        return filepath

# 단독 실행을 위한 코드
if __name__ == "__main__":
    search_queries = ["chatGPT", "Llama", "Gemini", "Grok", "Claude", "Midjourney", "CLOVA X", "Microsoft Copilot", "Devin", "GitHub Copilot", "뤼튼", "Stable Diffusion", "Runway", "Lumiere", "Sora", "kling", "Suno", "Udio"]  # 검색할 쿼리 설정
    crawler = NewsCrawler(search_queries)  # 크롤러 인스턴스 생성
    crawler.crawl_news(1, 20)  # 1페이지부터 20페이지까지 크롤링
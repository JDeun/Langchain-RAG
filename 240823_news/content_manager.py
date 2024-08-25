import os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import CONTENT_SAVE_DIR, RECENT_NEWS_DATE
from utils import parse_date
import re

class ContentManager:
    def __init__(self, search_queries):
        self.search_queries = search_queries
        self.contents = []
        self.filename = 'saved_newsletter.md'
        self.vectorizer = TfidfVectorizer()

    def save_content(self, content, filename=None):
        if filename is None:
            current_date = datetime.now().strftime("%y%m%d")
            filename = f"{current_date}_AI_newsletter.md"
        
        filepath = os.path.join(CONTENT_SAVE_DIR, filename)
        os.makedirs(CONTENT_SAVE_DIR, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"생성된 콘텐츠가 {filepath}에 저장되었습니다.")

    def load_all_contents(self):
        contents = []
        for filename in os.listdir(CONTENT_SAVE_DIR):
            if filename.endswith('.md'):
                filepath = os.path.join(CONTENT_SAVE_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    contents.append(file.read())
        return contents

    def check_similarity(self, new_content, threshold=0.7):
        existing_contents = self.load_all_contents()
        if not existing_contents:
            return False

        # 최근 1주일 내의 콘텐츠만 비교
        recent_contents = []
        for content in existing_contents:
            date_match = re.search(r'\d{6}_AI_newsletter\.txt', content)
            if date_match:
                date_str = date_match.group()[:6]
                parsed_date = parse_date(date_str)
                if parsed_date and parsed_date >= RECENT_NEWS_DATE:
                    recent_contents.append(content)

        if not recent_contents:
            return False

        all_contents = recent_contents + [new_content]
        tfidf_matrix = self.vectorizer.fit_transform(all_contents)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        return any(similarity > threshold for similarity in cosine_similarities)

    def get_unique_content(self, new_content, max_attempts=5):
        attempts = 0
        while attempts < max_attempts:
            if not self.check_similarity(new_content):
                return new_content
            attempts += 1
            # 여기서 새로운 콘텐츠를 생성하는 로직 호출 (예: newsletter_generator.generate_content())
            # new_content = newsletter_generator.generate_content(...)
        return None  # 최대 시도 횟수를 초과하면 None 반환
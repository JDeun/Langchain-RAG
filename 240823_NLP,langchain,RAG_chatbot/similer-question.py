import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from konlpy.tag import Okt

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# KoNLPy 형태소 분석기 초기화
okt = Okt()

def preprocess_text(text):
    """텍스트 전처리 함수"""
    # 형태소 분석 수행
    morphemes = okt.morphs(text, stem=True)
    # 불용어 제거 (예시, 필요에 따라 수정)
    stop_words = set(['은', '는', '이', '가', '을', '를', '들', '에', '의', '로'])
    return ' '.join([word for word in morphemes if word not in stop_words])

def load_questions(file_path):
    """파일에서 질문들을 로드하고 전처리하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file if line.strip()]
    return [preprocess_text(q) for q in questions]

def calculate_similarity(user_question, stored_questions):
    """TF-IDF와 코사인 유사도를 사용하여 유사도를 계산하는 함수"""
    # 사용자 질문 전처리
    preprocessed_user_question = preprocess_text(user_question)
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    all_questions = [preprocessed_user_question] + stored_questions
    tfidf_matrix = vectorizer.fit_transform(all_questions)

    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    return cosine_similarities

def find_most_similar_question(user_question, file_path):
    """가장 유사한 질문을 찾는 함수"""
    stored_questions = load_questions(file_path)
    original_questions = load_questions(file_path)  # 원본 질문 저장
    similarities = calculate_similarity(user_question, stored_questions)

    # 유사도 로깅
    for q, sim in zip(original_questions, similarities):
        logging.info(f"질문: '{q}', 유사도: {sim:.4f}")

    # 가장 유사한 질문 찾기
    most_similar_index = np.argmax(similarities)
    most_similar_question = original_questions[most_similar_index]
    highest_similarity = similarities[most_similar_index]

    logging.info(f"가장 유사한 질문: '{most_similar_question}', 유사도: {highest_similarity:.4f}")

    return most_similar_question, highest_similarity

# 메인 실행 부분
if __name__ == "__main__":
    file_path = r"C:\Users\gadi2\OneDrive\바탕 화면\study file\NLP_chatbot\queries_ko.txt"
    
    while True:
        user_input = input("질문을 입력하세요 (종료하려면 '종료' 입력): ")
        if user_input.lower() == '종료':
            break

        most_similar, similarity = find_most_similar_question(user_input, file_path)
        print(f"\n입력한 질문과 가장 유사한 질문: '{most_similar}' (유사도: {similarity:.4f})\n")

print("프로그램을 종료합니다.")
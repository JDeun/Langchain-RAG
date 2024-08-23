import os
import shutil
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from konlpy.tag import Okt
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 전역 변수 설정
DOCUMENT_PATH = r'C:\Users\gadi2\OneDrive\바탕 화면\study file\NLP_chatbot\rag_data.txt'
QUERIES_PATH = r'C:\Users\gadi2\OneDrive\바탕 화면\study file\NLP_chatbot\queries_ko.txt'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = "llama3.1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 2

# KoNLPy 형태소 분석기 초기화
okt = Okt()

def preprocess_text(text):
    """텍스트 전처리 함수"""
    morphemes = okt.morphs(text, stem=True)
    stop_words = set(['은', '는', '이', '가', '을', '를', '들', '에', '의', '로'])
    return ' '.join([word for word in morphemes if word not in stop_words])

def load_questions(file_path):
    """파일에서 질문들을 로드하고 전처리하는 함수"""
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [line.strip() for line in file if line.strip()]
    return questions, [preprocess_text(q) for q in questions]

def calculate_similarity(user_question, stored_questions):
    """TF-IDF와 코사인 유사도를 사용하여 유사도를 계산하는 함수"""
    preprocessed_user_question = preprocess_text(user_question)
    vectorizer = TfidfVectorizer()
    all_questions = [preprocessed_user_question] + stored_questions
    tfidf_matrix = vectorizer.fit_transform(all_questions)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

def find_most_similar_question(user_question, file_path):
    """가장 유사한 질문을 찾는 함수"""
    original_questions, stored_questions = load_questions(file_path)
    similarities = calculate_similarity(user_question, stored_questions)
    most_similar_index = np.argmax(similarities)
    most_similar_question = original_questions[most_similar_index]
    highest_similarity = similarities[most_similar_index]
    logging.info(f"입력: '{user_question}', 가장 유사한 질문: '{most_similar_question}', 유사도: {highest_similarity:.4f}")
    return most_similar_question, highest_similarity

async def setup_vectorstore(vectorstore_path):
    """벡터 저장소를 설정하는 비동기 함수"""
    if os.path.exists(vectorstore_path):
        try:
            shutil.rmtree(vectorstore_path)
            print("기존 벡터 저장소 삭제 완료")
        except PermissionError:
            print("기존 벡터 저장소를 삭제할 수 없습니다. 새로운 이름으로 저장소를 생성합니다.")
            vectorstore_path = 'vectorstore_new'

    loader = TextLoader(DOCUMENT_PATH)
    data = await asyncio.to_thread(loader.load)
    print("문서 로드 완료")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = await asyncio.to_thread(text_splitter.split_documents, data)
    print("문서 분할 완료")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore = await asyncio.to_thread(Chroma.from_documents, docs, embeddings, persist_directory=vectorstore_path)
    await asyncio.to_thread(vectorstore.persist)
    print(f"새 벡터 저장소 생성 및 저장 완료 (경로: {vectorstore_path})")

    return vectorstore, vectorstore_path

def setup_rag_chain(vectorstore):
    """RAG 체인을 설정하는 함수"""
    retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})
    llm = Ollama(model=LLM_MODEL)
    print("LLM 모델 로드 완료")

    template = '''당신은 전문적이고 친절한 고객 서비스 챗봇입니다. 주어진 컨텍스트를 바탕으로 고객의 질문에 정중하고 도움이 되는 방식으로 답변해 주세요. 
    항상 다음 지침을 따라주세요:

    1. 고객의 질문을 주의 깊게 이해하고, 관련된 정보만 제공하세요.
    2. 친절하고 공감하는 톤을 유지하며, 필요한 경우 고객의 상황에 대해 이해를 표현하세요.
    3. 명확하고 간결한 언어를 사용하여 정보를 전달하세요.
    4. 추가 질문이나 설명이 필요할 수 있음을 암시하고, 고객이 더 문의할 수 있도록 독려하세요.
    5. 정확한 정보를 제공할 수 없는 경우, 솔직히 인정하고 추가 도움을 받을 수 있는 방법을 안내하세요.
    6. 항상 예의 바르고 전문적인 태도를 유지하세요.

    컨텍스트:
    {context}

    고객 질문: {question}

    챗봇 답변:'''

    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    rag_chain = (
        {'context': retriever | format_docs, 'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

async def main():
    """메인 실행 함수"""
    vectorstore_path = 'vectorstore'
    vectorstore, vectorstore_path = await setup_vectorstore(vectorstore_path)
    rag_chain = setup_rag_chain(vectorstore)

    print("질문을 입력하세요. 종료하려면 '종료'를 입력하세요.")
    while True:
        user_input = input("질문: ")
        if user_input.lower() == '종료':
            break
        
        # 유사도 계산 및 가장 유사한 질문 찾기
        similar_question, similarity = find_most_similar_question(user_input, QUERIES_PATH)
        print(f"가장 유사한 질문: '{similar_question}' (유사도: {similarity:.4f})")

        # RAG 체인을 통한 답변 생성
        answer = await asyncio.to_thread(rag_chain.invoke, similar_question)
        print("답변:", answer)
        print("\n")

    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    asyncio.run(main())
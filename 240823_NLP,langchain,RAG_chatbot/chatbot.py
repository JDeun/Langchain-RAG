import os
import shutil
import asyncio
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 전역 변수 설정
DOCUMENT_PATH = r'C:\Users\gadi2\OneDrive\바탕 화면\study file\NLP_chatbot\rag_data.txt'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = "llama3.1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVER_K = 2

async def setup_vectorstore(vectorstore_path):
    """벡터 저장소를 설정하는 비동기 함수"""
    # 기존 벡터 저장소 삭제 시도
    if os.path.exists(vectorstore_path):
        try:
            shutil.rmtree(vectorstore_path)
            print("기존 벡터 저장소 삭제 완료")
        except PermissionError:
            print("기존 벡터 저장소를 삭제할 수 없습니다. 새로운 이름으로 저장소를 생성합니다.")
            vectorstore_path = 'vectorstore_new'

    # 문서 로드 및 분할
    loader = TextLoader(DOCUMENT_PATH)
    data = await asyncio.to_thread(loader.load)
    print("문서 로드 완료")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = await asyncio.to_thread(text_splitter.split_documents, data)
    print("문서 분할 완료")

    # 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
    )

    # 벡터 저장소 생성 및 저장
    os.makedirs(vectorstore_path, exist_ok=True)
    vectorstore = await asyncio.to_thread(Chroma.from_documents, docs, embeddings, persist_directory=vectorstore_path)
    await asyncio.to_thread(vectorstore.persist)
    print(f"새 벡터 저장소 생성 및 저장 완료 (경로: {vectorstore_path})")

    return vectorstore, vectorstore_path

def setup_rag_chain(vectorstore):
    """RAG 체인을 설정하는 함수"""
    # Retriever 설정
    retriever = vectorstore.as_retriever(search_kwargs={'k': RETRIEVER_K})

    # LLM 모델 설정
    llm = Ollama(model=LLM_MODEL)
    print("LLM 모델 로드 완료")

    # 프롬프트 템플릿 설정
    template = '''다음 컨텍스트를 바탕으로 질문에 간단히 답하세요:
    {context}
    질문: {question}
    답변:'''
    prompt = ChatPromptTemplate.from_template(template)

    # 문서 포맷팅 함수 정의
    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    # RAG Chain 구성
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
        query = input("질문: ")
        if query.lower() == '종료':
            break
        answer = await asyncio.to_thread(rag_chain.invoke, query)
        print("답변:", answer)
        print("\n")

    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    asyncio.run(main())
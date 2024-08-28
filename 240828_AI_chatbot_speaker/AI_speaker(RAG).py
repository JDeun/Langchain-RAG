import time
import os
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
from langchain.document_loaders import (
    DirectoryLoader, TextLoader, PDFMinerLoader, 
    Docx2txtLoader, CSVLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, UnstructuredODTLoader,
    UnstructuredPowerPointLoader, UnstructuredEPubLoader,
    UnstructuredImageLoader, UnstructuredEmailLoader,
    JSONLoader, UnstructuredRTFLoader, UnstructuredXMLLoader,
    EverNoteLoader, UnstructuredExcelLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
# from langchain.llms import OpenAI  # OpenAI GPT 주석 처리

# 환경 변수 설정 (OpenAI API 키)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 전역 변수로 종료 플래그 설정
is_running = True

# 데이터 로딩 및 처리
def load_and_process_data(directory_path):
    loaders = {
        ".txt": (TextLoader, {"encoding": "utf8"}),
        ".pdf": (PDFMinerLoader, {}),
        ".docx": (Docx2txtLoader, {}),
        ".doc": (Docx2txtLoader, {}),
        ".csv": (CSVLoader, {}),
        ".html": (UnstructuredHTMLLoader, {}),
        ".htm": (UnstructuredHTMLLoader, {}),
        ".md": (UnstructuredMarkdownLoader, {}),
        ".odt": (UnstructuredODTLoader, {}),
        ".pptx": (UnstructuredPowerPointLoader, {}),
        ".ppt": (UnstructuredPowerPointLoader, {}),
        ".epub": (UnstructuredEPubLoader, {}),
        ".jpg": (UnstructuredImageLoader, {}),
        ".jpeg": (UnstructuredImageLoader, {}),
        ".png": (UnstructuredImageLoader, {}),
        ".eml": (UnstructuredEmailLoader, {}),
        ".json": (JSONLoader, {"jq_schema": "."}),
        ".rtf": (UnstructuredRTFLoader, {}),
        ".xml": (UnstructuredXMLLoader, {}),
        ".enex": (EverNoteLoader, {}),
        ".xlsx": (UnstructuredExcelLoader, {}),
        ".xls": (UnstructuredExcelLoader, {}),
    }
    
    documents = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in loaders:
                loader_class, loader_args = loaders[file_extension]
                loader = loader_class(file_path, **loader_args)
                try:
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(texts, embeddings)
    
    return vectorstore

# RAG 모델 초기화(챗gpt의 경우)
# def initialize_rag(vectorstore):
#     llm = OpenAI(temperature=0)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vectorstore.as_retriever(),
#         return_source_documents=True
#     )
#     return qa_chain

# RAG 모델 초기화
def initialize_rag(vectorstore, llm):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa_chain

# 음성 인식 (듣기, STT)
def listen(recognizer, audio):
    try:
        text = recognizer.recognize_google(audio, language='ko')
        print('[사용자] ' + text)
        answer(text)
    except sr.UnknownValueError:
        print('인식 실패')  # 음성 인식 실패한 경우
        speak('죄송합니다. 다시 말씀해 주시겠어요?')
    except sr.RequestError as e:
        print('요청 실패: {0}'.format(e))  # API 키 오류, 네트워크 단절 등
        speak('죄송합니다. 음성 인식 서비스에 문제가 있습니다. 잠시 후 다시 시도해 주세요.')

# 대답 (RAG를 사용한 응답 생성)
def answer(input_text):
    global is_running
    if '종료' in input_text:
        answer_text = '네, 프로그램을 종료합니다. 이용해 주셔서 감사합니다.'
        speak(answer_text)
        is_running = False
        return

    # RAG를 사용한 응답 생성
    result = qa_chain({"query": input_text})
    answer_text = result['result']
    
    speak(answer_text)

# 소리내어 읽기 (TTS)
def speak(text):
    print('[인공지능] ' + text)
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)
    playsound(file_name)
    if os.path.exists(file_name):  # voice.mp3 파일 삭제
        os.remove(file_name)

# 메인 실행 부분
if __name__ == "__main__":
    # ChatOllama 초기화
    llm = ChatOllama(model="llama3.1")  # 또는 사용 가능한 다른 모델명
    # llm = OpenAI(model_name="gpt-4o-mini", temperature=0)

    # 데이터 로딩 및 RAG 모델 초기화
    directory_path = "path/to/your/data/directory" # load_and_process_data가 작동할 데이터가 있는 폴더
    vectorstore = load_and_process_data(directory_path)
    qa_chain = initialize_rag(vectorstore, llm)

    # 음성 인식 초기화
    r = sr.Recognizer()
    m = sr.Microphone()

    speak('안녕하세요. 무엇을 도와드릴까요? 종료하시려면 "종료"라고 말씀해 주세요.')
    
    with m as source:
        r.adjust_for_ambient_noise(source)  # 주변 소음에 맞게 마이크 조정
    
    stop_listening = r.listen_in_background(m, listen)

    try:
        while is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("키보드 인터럽트로 프로그램을 종료합니다.")
    finally:
        stop_listening(wait_for_stop=False)
        print("프로그램이 종료되었습니다.")
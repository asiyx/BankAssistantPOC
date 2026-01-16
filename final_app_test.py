import openai
import os
import streamlit as st
from dotenv import load_dotenv
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain


def handle_userinput(user_question, language_code):
    conversation = st.session_state.conversation
    if conversation is not None:
        response = conversation({'question': user_question, 'chat_history': []})
        user_question_nativeLang = translate_text(user_question, target_language=language_code)
        message_text = translate_text(response['answer'], target_language=language_code)
        tts_file = text_to_speech(message_text, language_code)
        st.write(user_template.replace("{{MSG}}", user_question_nativeLang), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", message_text), unsafe_allow_html=True)
        st.audio(tts_file, format='audio/mp3')
    else:
        st.error("Conversation session is not initialized.")


# Speech recognition for Voice search:
def recognize_speech(language_code='en'):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... aaannnddd....:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio, language=language_code)
    return text

def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    translated_text = translation.text
    return translated_text

def text_to_speech(text, lang_code):
    tts = gTTS(text, lang=lang_code)
    tts_file = BytesIO()
    tts.write_to_fp(tts_file)
    tts_file.seek(0)
    return tts_file


# Map languages to their respective codes
language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn"
}

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat Your Personal Assistant",
                       page_icon=":bank:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat Your Personal Assistant :bank:")

    # Language selection
    language = st.selectbox("Select Language", ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Bengali"])
    language_code = language_codes[language]

    user_question = st.text_input("Ask a question about your documents:")

    # Voice search button
    if st.button("Voice Search"):
        with st.spinner("Recording..."):
            user_NativeLang_question = recognize_speech(language_code) # Question in native language
            user_question = translate_text(user_NativeLang_question, target_language='en') # Converting the native language question into English for the understanding to model

    if user_question:
        # Clear previous chat history before handling the new user input
        st.session_state.chat_history = []
        handle_userinput(user_question, language_code)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
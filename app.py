import os
import time

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "leave_rules.pdf"

st.set_page_config(page_title="Leave Rules Q&A Bot", page_icon="📄")

# 1. Get the API key from Streamlit Secrets (never written in the code)
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found. Add it in the app's Settings -> Secrets.")
        st.stop()


# 2. Build the database ONCE when the app starts (same logic as your Colab notebook)
@st.cache_resource(show_spinner="Preparing the leave rules database (about 1 minute)...")
def load_everything():
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    pages = PyPDFLoader(PDF_PATH).load()
    splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitters.split_documents(pages)

    vectorstore = Chroma(embedding_function=embeddings, collection_name="leave_rule")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        vectorstore.add_documents(chunks[i : i + batch_size])
        time.sleep(2)
    return llm, vectorstore


llm, vectorstore = load_everything()


# 3. Your ask() function (same logic as Colab, plus simple error handling)
def ask(question):
    if not question or not question.strip():
        return "Please type a question about leave rules."
    try:
        results = vectorstore.similarity_search(question, k=4)

        context = ""
        pages_used = []
        for r in results:
            context = context + r.page_content + "\n\n"
            pages_used.append(r.metadata["page"] + 1)

        prompt = "Answer ONLY from this context. If not found, say 'I could not find this in the document.'\n\n"
        prompt = prompt + context + "Question: " + question

        answer = llm.invoke(prompt).text
        return answer + "\n\nSource: pages " + str(sorted(set(pages_used)))
    except Exception as e:
        print("Error:", e)
        return "Sorry, something went wrong. Please try again in a moment."


# 4. The web page
st.title("📄 Leave Rules Q&A Bot")
st.write("Ask questions about the Central Civil Services (Leave) Rules, 1972.")

with st.form("question_form"):
    question = st.text_input("Ask a question about leave rules")
    submitted = st.form_submit_button("Get answer")

if submitted:
    with st.spinner("Searching the rules..."):
        st.write(ask(question))

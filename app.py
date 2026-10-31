import os
import time

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# the key now comes from Streamlit Secrets (not from Colab userdata)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

#  the PDF is inside the project (no files.upload)
pdf_name = "leave_rules.pdf"


#  your steps are put in one function so Streamlit builds the database only once
@st.cache_resource(show_spinner="Preparing the database (about 1 minute)...")
def build_database():
    loader = PyPDFLoader(pdf_name)
    pages = loader.load()

    splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitters.split_documents(pages)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstores = Chroma(embedding_function=embeddings, collection_name="leave_rule")
    batch_size = 20

    for i in range(0, len(chunks), batch_size):
        vectorstores.add_documents(chunks[i : i + batch_size])
        time.sleep(2)

    return vectorstores


vectorstore = build_database()


# Your ask() function, exactly the same
def ask(question):
    results = vectorstore.similarity_search(question, k=4)

    context = ""
    pages = []
    for r in results:
        context = context + r.page_content + "\n\n"
        pages.append(r.metadata["page"] + 1)

    prompt = "Answer ONLY from this context. If not found, say 'I could not find this in the document.'\n\n"
    prompt = prompt + context + "Question: " + question

    answer = llm.invoke(prompt).text
    return answer + "\n\n source : pages" + str(sorted(set(pages)))


# Streamlit page instead of Gradio
st.title("Leave Rule Q & A Bot")
st.write("Ask the question about leave rules")

question = st.text_input("Ask the question about leave rules")

if st.button("Get answer"):
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching..."):
            st.write(ask(question))

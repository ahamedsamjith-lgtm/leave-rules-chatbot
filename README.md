# Leave Rules Chatbot

A chatbot that answers questions from the Central Civil Services (Leave) Rules, 1972. You type a question, it finds the matching parts of the PDF and answers from them, and it tells you which pages it used.

Live demo: https://leave-rules-chatbot.streamlit.app

It is on a free plan, so if nobody has opened it for a while it can take a minute to wake up.

## About this project

This is a practice project I made to learn how RAG works. I first built it in Google Colab, then moved it into a normal app so anyone can use it with a link. I tried Hugging Face Spaces first, but it needed a paid plan for this kind of app, so I used Streamlit Community Cloud instead.

The PDF has 73 pages. The bot only answers from that document. If the answer isn't in it, it says so instead of guessing.

## How it works

1. The PDF is cut into chunks of about 1000 characters.
2. Each chunk is turned into an embedding with Google's gemini-embedding-001 and saved in Chroma.
3. When you ask something, the 4 closest chunks are pulled out.
4. Those chunks and your question go to Gemini (gemini-flash-lite-latest), with an instruction to answer only from them.
5. The answer is shown with the page numbers.

Tools used: Python, LangChain, Gemini, Chroma, PyPDF, Streamlit.

## Files

- app.py: the app
- requirements.txt: the libraries it needs
- leave-rules-chatbot.pdf: the rules document
- RAG_project.ipynb: my original Colab notebook

## Running it yourself

pip install -r requirements.txt
streamlit run app.py

You need your own Google API key. Create a file called .streamlit/secrets.toml and put this in it:

GOOGLE_API_KEY = "your key"

Don't upload that file to GitHub.
## Limits

It only knows this one PDF, and the answers depend on which chunks it finds. It's a learning project, so check the original rules before relying on an answer.

Ahamed Samjith

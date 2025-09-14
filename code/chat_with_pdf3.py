print("Hello suman....started....")

# pip3 install langchain langchain_community langchain_openai chromadb

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate

# --- Load API key ---
# load from .env for local development
load_dotenv()

print("Hello suman....started....one ")

# get from Streamlit Secrets first, fallback to .env
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it in Streamlit Cloud -> Manage app -> Secrets.")

# --- Create embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Vector store (Chroma) ---
vector_store = Chroma(
    collection_name="my_collection",
    persist_directory="./knowledge_base",
    embedding_function=embeddings.embed_query  # pass embedding function callable
)
print("Hello suman....started....two ")
# --- Prompt template ---
template = """
You are a helpful assistant answering all the user questions.
Answer the user questions based on the context provided.
If you do not know the answer, please say "Don't know". 
Do not fabricate the answer at any cost.

Question: {question}
Context: {context}
"""

# --- LLM ---
# Replace with a model you have access to
llm = ChatOpenAI(model="gpt-5-nano", openai_api_key=OPENAI_API_KEY)

print("Hello suman....started....three ")

# --- Streamlit UI ---
st.header("Chat with PDF")

# Input from user
question = st.chat_input("Enter your question> ")

if question:
    # search relevant docs
    try:
        docs = vector_store.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    except Exception as e:
        context = f"Vector store error: {e}"

    # build prompt
    prompt_template = ChatPromptTemplate.from_template(template=template)
    prompt = prompt_template.invoke({
        "question": question,
        "context": context
    })

    # LLM answer
    try:
        result = llm.invoke(prompt)
        st.text(result.content)
    except Exception as e:
        st.error(f"LLM error: {e}")

import os
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma, FAISS
from langchain.prompts import ChatPromptTemplate

# --- Load API key ---
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it in Streamlit Cloud -> Manage app -> Secrets.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Embeddings ---
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# --- Try Chroma, fallback to FAISS ---
try:
    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory="./knowledge_base",
        embedding_function=embeddings.embed_query
    )
    st.write("✅ Using Chroma as vector store")
except Exception as e:
    st.warning(f"Chroma failed ({e}). Falling back to FAISS.")
    vector_store = FAISS.from_texts([], embeddings)

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
llm = ChatOpenAI(model="gpt-5-nano", openai_api_key=OPENAI_API_KEY)

# --- Streamlit UI ---
st.header("Chat with PDF")

question = st.chat_input("Enter your question> ")

if question:
    try:
        docs = vector_store.similarity_search(question, k=5)
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found."
    except Exception as e:
        context = f"Vector store search error: {e}"

    prompt_template = ChatPromptTemplate.from_template(template=template)
    prompt = prompt_template.invoke({"question": question, "context": context})

    try:
        result = llm.invoke(prompt)
        st.text(result.content)
    except Exception as e:
        st.error(f"LLM error: {e}")

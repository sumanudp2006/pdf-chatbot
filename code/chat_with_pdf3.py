# pip3 install langchain langchain_community langchain_openai langchain_chroma 

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
import streamlit as st

# load all the keys from .env file
load_dotenv()

# create embeddings  / creates embeddings for text.
embeddings = OpenAIEmbeddings()

# suman: loads (or creates) a persistent vector DB stored under ./knowledge_base.
# load vector_store
vector_store = Chroma(
    collection_name="my_collection",
    persist_directory="./knowledge_base",
    embedding_function=embeddings
)

# create a prompt template
# template = """
# You are a helpful assistant answering all the user questions.
# Answer the user questions based on the context provided.
# If you do not know the answer, please use your existing knowledge. 
# Do not fabricate the answer at any cost.

# Question: {question}
# Context: {context}
# """

template = """
You are a helpful assistant answering all the user questions.
Answer the user questions based on the context provided.
If you do not know the answer, please say "Don't know". 
Do not fabricate the answer at any cost.

Question: {question}
Context: {context}
"""

# create llm
#llm = ChatOpenAI(model="gpt-5")
llm = ChatOpenAI(model="gpt-5-nano")

# set the streamlit UI
st.header("Chat with PDF")

# get the question from user
question = st.chat_input("enter your question> ")

# check if question is asked by user
if question:
    # suman: vector_store.search(question, ...) returns nearest documents from that DB; those are fed into the prompt template and the model answers
    # find the similar documents from vector_store
    context = vector_store.search(
        question, search_type="similarity", k=5
    )

    # create the prompt template
    prompt_template = ChatPromptTemplate.from_template(template=template)

    # set the variables and create the prompt
    prompt = prompt_template.invoke({
        "question": question,
        "context": context
    })

    # send the prompt and get the result
    result = llm.invoke(prompt)
    st.text(result.content)

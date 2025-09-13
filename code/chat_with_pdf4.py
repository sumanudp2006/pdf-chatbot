# pip3 install langchain langchain_community langchain_openai langchain_chroma

from flask import Flask, request
from flask import Flask, request, jsonify


from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate

# load all the keys from .env file
load_dotenv()

# create embeddings
embeddings = OpenAIEmbeddings()

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

# create a flask app
app = Flask(__name__)

@app.route("/ask", methods=['POST'])
def answer_question():
    # read the question from request object
    question = request.form.get("question")

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
    return result.content

app.run(host="0.0.0.0", port=6790, debug=True)
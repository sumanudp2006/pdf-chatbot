from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def create_knowledge_base():
    # create an embeddings object
    embeddings = OpenAIEmbeddings()

    # dummy data
    documents = ["this is text1", "this is text2"]

    # create embeddings for the dummy data
    dummy_embeddings = embeddings.embed_documents(documents)
    print(dummy_embeddings)

create_knowledge_base()
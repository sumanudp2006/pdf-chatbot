# pip3 install langchain langchain_community langchain_openai langchain_chroma 

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# load all the keys from .env file
load_dotenv()

def load_pdf_file():
    file = "resume.pdf"

    # create a loader
    loader = PyMuPDFLoader(file_path=file)
    
    # read the pages
    pages = loader.load()

    # collect all the contents in Document format
    documents = []

    # read all the pages and convert them into documents
    id = 1
    for page in pages:
        documents.append(
            Document(page_content=page.page_content, id=id)
        )
        id += 1

    return documents


def create_knowledge_base():
    # create an embeddings object
    openai_embeddings = OpenAIEmbeddings()
    # ollama_embeddings = OllamaEmbeddings(model="llama3.1")

    # read the contents of the required file
    print("reading the data from file")
    documents = load_pdf_file()

    # create a splitter to split the data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )

    # split the data into smaller chunks    
    chunks = splitter.split_documents(documents)

    # create vector store
    print(f"building the vector store")
    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory="./knowledge_base",
        embedding_function=openai_embeddings
    )

    # add the chunks to the vector store
    print("adding the chunks")
    vector_store.add_documents(chunks)

#create_knowledge_base()

def test_knowledge_base():
    # create an embeddings object
    openai_embeddings = OpenAIEmbeddings()
        
    # load the chroma database
    vector_store = Chroma(
        collection_name="my_collection",
        persist_directory="./knowledge_base",
        embedding_function=openai_embeddings
    )

    # find anything related to Allan
    similar_documents = vector_store.search(
        "allan kulkarni", 
        search_type="similarity",
        k=5
    )
    print(similar_documents)


test_knowledge_base()
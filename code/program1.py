# pip3 install langchain langchain_community langchain_openai langchain_chroma 

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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

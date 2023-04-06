from langchain.document_loaders import UnstructuredHTMLLoader as HTMLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from glob import glob
from tqdm.auto import tqdm
from .const import MODEL_NAME

def create_loaders():
    files = glob("pl-docs/*.html")
    return [HTMLLoader(file) for file in files]

def create_documents():
    documents = []
    ids = []
    files = glob("pl-docs/*.html")
    for file in tqdm(files):
        loader = HTMLLoader(file)
        document = loader.load()
        documents.extend(document)
        ids.append(file)
    return documents, ids


def create_db():
    documents, ids = create_documents()
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    db = Chroma.from_documents(documents, embeddings, ids=ids)
    return db, documents, embeddings

from glob import glob

from langchain.document_loaders import UnstructuredHTMLLoader as HTMLLoader
from langchain.vectorstores import Chroma
from tqdm.auto import tqdm

from qna.model import create_embedding

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
    embeddings = create_embedding()
    db = Chroma.from_documents(documents, embeddings, ids=ids)
    return db, documents, embeddings

from langchain.embeddings import HuggingFaceEmbeddings

from qna.const import MODEL_NAME


def create_embedding():
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return embeddings

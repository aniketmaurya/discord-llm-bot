from qna.db import create_db


def create_retriever():
    db, documents, embeddings = create_db()
    retriever = db.as_retriever()
    return retriever

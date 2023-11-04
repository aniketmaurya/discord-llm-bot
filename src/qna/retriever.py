from qna.db import create_db


def create_retriever():
    db, documents, embeddings = create_db()
    retriever = db.as_retriever()
    return retriever


class LightningRetriever:
    from .pl_db import get_collection, MODEL_NAME
    from chromadb.utils import embedding_functions

    collection = get_collection()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )

    def __call__(self, query: str):
        query_texts = [query]
        query_embeddings = self.sentence_transformer_ef(query_texts)
        result = self.collection.query(query_embeddings=query_embeddings, n_results=1)

        return {
            "document": result["documents"][0][0],
            "distance": result["distances"][0][0],
            "source": result["metadatas"][0][0]["source"],
        }

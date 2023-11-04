from chromadb.utils import embedding_functions

from .db import create_db
from .pl_db import MODEL_NAME, Document, get_collection, get_table

DEFAULT_DB_ENGINE = "lancedb"


def create_retriever():
    db, documents, embeddings = create_db()
    retriever = db.as_retriever()
    return retriever


class LightningRetriever:
    def __init__(self, engine_type: str = DEFAULT_DB_ENGINE):
        self.engine_type = engine_type
        if engine_type == "chromadb":
            self.collection = get_collection()
            self.sentence_transformer_ef = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=MODEL_NAME
                )
            )
        elif engine_type == "lancedb":
            self.table = get_table()

    def chroma_engine(self, query: str):
        query_texts = [query]
        query_embeddings = self.sentence_transformer_ef(query_texts)
        result = self.collection.query(query_embeddings=query_embeddings, n_results=1)

        return {
            "document": result["documents"][0][0],
            "distance": result["distances"][0][0],
            "source": result["metadatas"][0][0]["source"],
        }

    def lance_engine(self, query: str):
        result: Document = (
            self.table.search(query, vector_column_name="embedding")
            .limit(1)
            .to_list()[0]
        )
        return {
            "document": result["document"],
            "distance": result["_distance"],
            "source": result["source"],
        }

    def __call__(self, query: str):
        if self.engine_type == "lancedb":
            return self.lance_engine(query=query)
        else:
            return self.chroma_engine(query=query)

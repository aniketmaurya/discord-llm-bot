from glob import glob
from os.path import basename
from pathlib import Path

import chromadb
import lancedb
import pandas as pd
import torch
from chromadb.utils import embedding_functions
from lancedb.embeddings import EmbeddingFunctionRegistry
from lancedb.pydantic import LanceModel, Vector
from loguru import logger
from rich import print
from rich.progress import track

MODEL_NAME = "all-distilroberta-v1"
DB_PATH = "db/lancedb-test"
TABLE_NAME = COLLECTION_NAME = "test"

registry = EmbeddingFunctionRegistry.get_instance()
func = registry.get("sentence-transformers").create(
    name="all-distilroberta-v1", device="cuda" if torch.cuda.is_available() else "cpu"
)


class Document(LanceModel):
    document: str = func.SourceField()
    embedding: Vector(func.ndims()) = func.VectorField()
    source: str


def get_collection() -> chromadb.Collection:
    chroma_client = chromadb.PersistentClient(DB_PATH)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        logger.exception(e)
        logger.warning("Indexing documents...")
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
        csvs = glob("crawled/*.csv")
        sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=MODEL_NAME
            )
        )
        data = []
        for csv in track(csvs):
            df = pd.read_csv(csv)
            if len(df) == 0:
                continue
            urls, documents = df["URL"].tolist(), df["Section Content"].tolist()
            embeddings = sentence_transformer_ef(documents)
            assert len(urls) == len(documents) == len(embeddings)
            base = basename(urls[0])
            collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=[{"source": url} for url in urls],
                ids=[f"{base}_{i}" for i in range(len(documents))],
            )

    return collection


def get_table():
    uri = DB_PATH[:]
    db = lancedb.connect(uri)
    table = db.open_table(TABLE_NAME)
    return table

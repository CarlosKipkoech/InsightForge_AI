from langchain_qdrant import QdrantVectorStore

from src.rag.embeddings import get_embeddings

COLLECTION_NAME = "insightforge_documents"


def create_qdrant_store(chunks):

    """
    Creates a Qdrant vector store from the provided documents and embeddings.
    """
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        path="data/qdrant_db",
        collection_name="COLLECTION_NAME",
    )
    return vector_store



def load_qdrant_store():
    """
    Loads an existing Qdrant vector store from disk.
    """
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore(
        path="data/qdrant_db",
        collection_name="COLLECTION_NAME",
        embedding=embeddings,
    )
    return vector_store
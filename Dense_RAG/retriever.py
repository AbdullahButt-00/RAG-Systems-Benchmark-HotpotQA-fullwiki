"""
Dense retriever backed by sentence-transformers/multi-qa-MiniLM-L6-cos-v1 and FAISS,
integrated through LlamaIndex's VectorStoreIndex abstraction.
"""

import faiss

from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from config import (
    EMBED_MODEL_NAME,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
    FAISS_INDEX_FILE,
    STORAGE_DIR,
    TOP_K,
)


def _configure_settings() -> None:
    """Apply global LlamaIndex settings. Safe to call multiple times."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    Settings.llm = None
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 0
    Settings.transformations = [SentenceSplitter(chunk_size=1024, chunk_overlap=0)]


class DenseRetriever:
    """
    Wraps a LlamaIndex VectorStoreIndex backed by a FAISS flat L2 index.

    Typical usage
    -------------
    First run (build once):
        retriever = DenseRetriever()
        retriever.build_index(documents)   # documents: List[llama_index Document]

    Subsequent runs (load from disk):
        retriever = DenseRetriever()
        retriever.load_index()

    Retrieval:
        passages = retriever.retrieve("What is the capital of France?", top_k=5)
    """

    def __init__(self) -> None:
        _configure_settings()
        self._index: VectorStoreIndex | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, documents: list) -> None:
        """
        Encode all documents and build a FAISS flat L2 index.
        Persists both the LlamaIndex storage context and the raw FAISS binary.
        """
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

        print(f"Building FAISS index over {len(documents):,} passages...")
        faiss_index = faiss.IndexFlatL2(EMBED_DIM)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True,
        )

        # Persist LlamaIndex docstore / index store
        self._index.storage_context.persist(persist_dir=str(STORAGE_DIR))
        # Persist raw FAISS binary separately (required for FaissVectorStore reload)
        vector_store.persist(persist_path=str(FAISS_INDEX_FILE))

        print(f"Index persisted → storage: {STORAGE_DIR}, faiss: {FAISS_INDEX_FILE}")

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Load a previously persisted index from disk."""
        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_FILE}.\n"
                "Build it first by running:  python preprocess.py"
            )
        if not STORAGE_DIR.exists():
            raise FileNotFoundError(
                f"LlamaIndex storage not found at {STORAGE_DIR}.\n"
                "Build it first by running:  python preprocess.py"
            )

        print("Loading index from disk...")
        vector_store = FaissVectorStore.from_persist_path(str(FAISS_INDEX_FILE))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(STORAGE_DIR),
        )
        self._index = load_index_from_storage(storage_context)
        print("Index loaded.")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Retrieve top_k passages most similar to query.

        Returns
        -------
        List of dicts, each with keys:
            passage_id (str), title (str), text (str), score (float)
        """
        if self._index is None:
            raise RuntimeError(
                "Index not initialised. Call load_index() or build_index() first."
            )
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        return [
            {
                "passage_id": n.node.metadata.get("passage_id", ""),
                "title": n.node.metadata.get("title", ""),
                "text": n.node.text,
                "score": float(n.score if n.score is not None else 0.0),
            }
            for n in nodes
        ]

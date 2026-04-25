"""
Dense seed retriever — LlamaIndex VectorStoreIndex backed by FAISS.
Acts as the first stage (seed retrieval) in the graph-guided pipeline.
"""

import gc
import json

import faiss
from pathlib import Path
from tqdm import tqdm

from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core import load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from config import (
    EMBED_MODEL_NAME,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
    INDEX_BATCH_SIZE,
    FAISS_INDEX_FILE,
    STORAGE_DIR,
    TOP_K_SEED,
)


def _configure_settings() -> None:
    """Apply global LlamaIndex settings. Safe to call multiple times."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=EMBED_BATCH_SIZE,
    )
    Settings.llm = None
    # chunk_size=512: safely above the longest HotpotQA passage (~300 tokens)
    # with lightweight metadata (title + passage_id only, ~20 tokens).
    # Prevents splitting while keeping peak RAM low on CPU.
    Settings.chunk_size = 512
    Settings.chunk_overlap = 0
    Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=0)]


class DenseRetriever:
    """
    Wraps a LlamaIndex VectorStoreIndex backed by a FAISS flat L2 index.

    Typical usage
    -------------
    First run (build once):
        retriever = DenseRetriever()
        retriever.build_index(documents)

    Subsequent runs (load from disk):
        retriever = DenseRetriever()
        retriever.load_index()

    Seed retrieval:
        seeds = retriever.retrieve_seed("Who directed Inception?", top_k=5)
    """

    def __init__(self) -> None:
        _configure_settings()
        self._index: VectorStoreIndex | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, corpus_path: Path) -> None:
        """
        Build and persist FAISS index from corpus.json using incremental batching.

        Processes INDEX_BATCH_SIZE documents at a time so peak RAM never holds
        more than one batch of nodes + the running docstore accumulation.
        The full document list is never materialised in memory.

        Parameters
        ----------
        corpus_path : Path
            Path to corpus.json produced by preprocess.build_corpus().
        """
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        FAISS_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Load the raw dict once (~600 MB) — far cheaper than 508k Document objects.
        print("Loading corpus for indexing...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus: dict = json.load(f)
        pids = list(corpus.keys())
        total = len(pids)
        print(f"Building FAISS index over {total:,} passages "
              f"(batch={INDEX_BATCH_SIZE}, embed_batch={EMBED_BATCH_SIZE})...")

        # Empty index — nodes are inserted batch-by-batch below.
        raw_faiss = faiss.IndexFlatL2(EMBED_DIM)
        vector_store = FaissVectorStore(faiss_index=raw_faiss)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self._index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=0)

        for batch_start in tqdm(
            range(0, total, INDEX_BATCH_SIZE),
            desc="Indexing batches",
            unit="batch",
        ):
            batch_pids = pids[batch_start : batch_start + INDEX_BATCH_SIZE]

            # Build Document objects for this batch only — released after insert.
            docs = [
                Document(
                    text=corpus[pid]["text"],
                    doc_id=pid,
                    metadata={"title": corpus[pid]["title"], "passage_id": pid},
                )
                for pid in batch_pids
            ]

            # Parse → embed → add to FAISS + docstore, all within this batch.
            nodes = node_parser.get_nodes_from_documents(docs, show_progress=False)
            self._index.insert_nodes(nodes)

            # Explicitly drop batch objects so GC can reclaim before next iteration.
            del docs, nodes, batch_pids
            gc.collect()

        del corpus, pids
        gc.collect()

        self._index.storage_context.persist(persist_dir=str(STORAGE_DIR))
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
        print("Loading FAISS index from disk...")
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

    def retrieve_seed(self, query: str, top_k: int = TOP_K_SEED) -> list[dict]:
        """
        Return top_k dense matches for the query.
        These are the seed passages for downstream graph expansion.

        Returns list of dicts with keys:
            passage_id, title, text, sentences, score
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
                # sentences not in node metadata — graph_retriever attaches
                # them from corpus.json lookup after BFS expansion.
                "score": float(n.score if n.score is not None else 0.0),
            }
            for n in nodes
        ]

    def retrieve(self, query: str, top_k: int = TOP_K_SEED) -> list[dict]:
        """Backward-compatible alias for retrieve_seed()."""
        return self.retrieve_seed(query, top_k)

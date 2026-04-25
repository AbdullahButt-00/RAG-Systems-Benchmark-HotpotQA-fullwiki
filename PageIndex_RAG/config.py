from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# Dataset — shared with Dense_RAG and Graph_RAG (one level up)
DATASET_DIR = PROJECT_ROOT.parent / "DATASET"

# Processed corpus
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.json"
PASSAGE_LOOKUP_PATH = DATA_DIR / "passage_lookup.json"

# PageIndex document bundles (plain-text files submitted to API)
PAGEINDEX_DOCS_DIR = PROJECT_ROOT / "pageindex_docs"

# PageIndex index persistence
PAGEINDEX_INDEX_DIR = PROJECT_ROOT / "pageindex_index"
DOC_ID_PERSIST_PATH = PAGEINDEX_INDEX_DIR / "doc_ids.json"

# PageIndex API
PAGEINDEX_API_KEY = "901921080abe41f7b6cd8579c152f151"
PAGEINDEX_POLL_INTERVAL = 5          # seconds between status polls
PAGEINDEX_ENABLE_CITATIONS = True
PAGEINDEX_TEMPERATURE = 0

# Corpus limit (set to None for full dataset)
CORPUS_MAX_PASSAGES = 50_000

# Ingestion
PASSAGES_PER_DOCUMENT = 250           # passages bundled per submitted document

# Pre-filter (Option B)
MAX_CANDIDATE_DOCS = 10              # max doc_ids queried per question

# Evaluation
EVAL_SAMPLE_SIZE = 500
EVAL_REPORT_PATH = PROJECT_ROOT / "evaluation_report.json"

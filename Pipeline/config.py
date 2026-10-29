# ======================================================================================
# --- config.py: GENERAL CONTROL PANEL (SETTINGS AND IMPORTS) ---
# ======================================================================================
import os
import torch

print("Loading core configurations...")

# --- 1. PATHS AND FILES ---
_DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_FOLDER = os.environ.get("BATTERIES_PROJECT_FOLDER", _DEFAULT_ROOT)

DATA_FOLDER = os.path.join(PROJECT_FOLDER, "Data")
if not os.path.isdir(DATA_FOLDER) and os.path.isdir(os.path.join(PROJECT_FOLDER, "01_dados")):
    DATA_FOLDER = os.path.join(PROJECT_FOLDER, "01_dados")

RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "Results")
if not os.path.isdir(RESULTS_FOLDER) and os.path.isdir(os.path.join(PROJECT_FOLDER, "04_resultados")):
    RESULTS_FOLDER = os.path.join(PROJECT_FOLDER, "04_resultados")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

DATA_FILE_PATH = os.path.join(DATA_FOLDER, "df_data_batteries_130k.parquet")
FINAL_CORPUS_CSV_PATH = os.path.join(DATA_FOLDER, "final_corpus_data.csv")
TOPIC_RELEVANCE_INDEX_FILE_PATH = os.path.join(DATA_FOLDER, "topic_relevance_index.parquet")
TOPIC_RELEVANCE_INDEX_CSV_PATH = os.path.join(DATA_FOLDER, "topic_relevance_index.csv")
DATA_PHRASES_FILE_PATH = os.path.join(DATA_FOLDER, "df_data_phrases.parquet")
DATA_PHRASES_CSV_PATH = os.path.join(DATA_FOLDER, "df_data_phrases.csv")

# --- 2. HARDWARE DETECTION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_THRESHOLD_GB = 8.0
IS_PRODUCTION_ENV = False

if DEVICE == "cuda":
    try:
        gpu_properties = torch.cuda.get_device_properties(0)
        gpu_vram_gb = gpu_properties.total_memory / (1024**3)
        print(f"GPU detected: {gpu_properties.name} with {gpu_vram_gb:.2f} GB VRAM")
        IS_PRODUCTION_ENV = True
    except Exception as e:
        print(f"Could not retrieve GPU properties, using CPU. Error: {e}")
        DEVICE = "cpu"
else:
    print("No GPU detected. Using CPU settings.")

# --- 3. DEFINITION OF ALL MODELS ---
EMBEDDING_MODEL_TEST = "all-MiniLM-L6-v2"    # Dimension: 384
EMBEDDING_MODEL_PROD = "all-mpnet-base-v2"  # Dimension: 768

VLLM_MODEL_NAME = "RedHatAI/gemma-3-4b-it-quantized.w8a8"
LLM_CLASSIFIER_MODEL = VLLM_MODEL_NAME
LLM_ADVANCED_MODEL = VLLM_MODEL_NAME

# --- 4. AUTOMATIC MODEL SELECTION FOR THIS SESSION ---
EMBEDDING_MODEL = EMBEDDING_MODEL_PROD if IS_PRODUCTION_ENV else EMBEDDING_MODEL_TEST

EMBEDDING_MODEL_SAFE_NAME = EMBEDDING_MODEL.replace("/", "_")
LLM_CLASSIFIER_MODEL_SAFE_NAME = LLM_CLASSIFIER_MODEL.replace("/", "_").replace(":", "_")
LLM_ADVANCED_MODEL_SAFE_NAME = LLM_ADVANCED_MODEL.replace("/", "_").replace(":", "_")

BATCH_SIZE_EMBEDDING = 128 if IS_PRODUCTION_ENV else 32

# --- 6. VLLM SETTINGS ---
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "not-needed")

# --- 7. Column Mapping ---
COLUMN_DOI = "DOI"
COLUMN_TITLE = "Document Title"
COLUMN_ABSTRACT = "Abstract"
COLUMN_AUTHORS_LIST = "authors_list"
COLUMN_AUTHORS_FULL_NAME_LIST = "authors_full_name_list"
COLUMN_AUTHORS_INSTITUTION = "authors_institution"
COLUMN_AUTHORS_COUNTRY = "authors_country"
COLUMN_YEAR = "Year Published"
COLUMN_DOCUMENT_TYPE = "Document Type"
COLUMN_AUTHORS_KEYWORDS = "Author Keywords"
COLUMN_AUTHORS_KEYWORDS_NORM_LIST = "author_keywords_normalized_list"
COLUMN_FIRST_AUTHOR = "first_author"
COLUMN_LAST_AUTHOR = "last_author"
COLUMN_FIRST_AUTHOR_FULL_NAME = "first_author_full_name"
COLUMN_LAST_AUTHOR_FULL_NAME = "last_author_full_name"
COLUMN_FIRST_INSTITUTION = "first_institution"
COLUMN_LAST_INSTITUTION = "last_institution"
COLUMN_FIRST_COUNTRY = "first_country"
COLUMN_LAST_COUNTRY = "last_country"
COLUMN_FIRST_COUNTRY_REGION = "first_country_region"
COLUMN_LAST_COUNTRY_REGION = "last_country_region"
COLUMN_TOPIC_NAME = "topic_cluster_1"
COLUMN_CITATIONS_COUNT = "Total Times Cited Count"
COLUMN_CITATIONS_PER_YEAR = "citations_per_year"
COLUMN_JOURNAL = "Publication Name"
COLUMN_JOURNAL_IF = "journal_if"
COLUMN_TOPIC_RANK_INDEX = "topic_rank_index"
COLUMN_TOPIC_RANK_INDEX_POSITION = "topic_rank_index_position"
COLUMN_TOPIC_RANK_INDEX_WO_REVIEW = "topic_rank_index_wo_review"
COLUMN_TOPIC_RANK_INDEX_WO_REVIEW_POSITION = "topic_rank_index_wo_review_position"
COLUMN_RESEARCH_TYPE = "research_type"
COLUMN_CATEGORY_GROUP = "category_group"
COLUMN_JNIF = "JNIF"
COLUMN_LANGUAGE = "Language"
COLUMN_WOS_CATEGORIES = "WOS Categories"
COLUMN_WOS_RESEARCH_AREAS = "Research Areas"

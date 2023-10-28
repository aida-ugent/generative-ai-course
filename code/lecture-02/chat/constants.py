from pathlib import Path

# variables for prompt
EMBEDDING_PROMPT = "passage: {text}"
QUERY_PROMPT = "query: {query}"

CHAT_PROMPT_TEMPLATE = """"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {user_msg}
ASSISTANT:"""

CONTEXT_PROMPT_TEMPLATE  = (
    "context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "Given the context information and not prior knowledge, answer the following query, only use the relevant context information.\n"
    "---------------------\n"
    "{query_str}\n"
    "---------------------\n"
)

# variables for preprocessing

PROCESSED_DATA_FILE = "df_processed_data.parquet"
CHUNKED_FILE = "df_processed_data_chunked.parquet"
CHUNKED_EMB_FILE = "df_processed_data_chunked_emb.parquet"
CHUNK_OVERLAP = 20


# variables for server
API_HOST = "localhost"
API_PORT = 8081
WORKER_ADDR = f"http://{API_HOST}:{API_PORT}"

# variables for embeddings
DATA_FOLDER = Path("../tmp") / "data"
EMBEDDING_MODEL_REPO_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_MODEL_DEVICE = "cuda:0"

# variables for ranking
RETRIEVAL_TOP_K = 5


# variables for chainlit options
CHAT_MODES = ["Plain", "RAG"]
LLMS = ["vicuna-13b-v1.5"]

# variables for chat model
CHAT_MODEL_REPO_NAME = "lmsys/vicuna-7b-v1.5"
MAX_LENGTH = 4096
# MODEL_TEMPERATURE = 0


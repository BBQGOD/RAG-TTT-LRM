import os
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm import bm25_embedding, openai_complete_if_cache
from lightrag.utils import MODEL, EmbeddingFunc

WORKING_DIR = "./working_dir"
MAX_WORKERS = 16
BASE_URL = "http://localhost:7689/v1"
API_KEY = "token-abc123"
TEMPERATURE = 0
MAX_TOKENS = 20480 # 4096, 8192, 20480

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def vllm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    _ = kwargs.pop("keyword_extraction", None)
    return await openai_complete_if_cache(
        MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        **kwargs,
    )

rag = LightRAG(
    llm_model_func=vllm_complete,
    llm_model_max_async=MAX_WORKERS,
    llm_model_max_token_size=32768,
    vector_storage="BM25VectorDBStorage",
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=2048,
        func=lambda texts: bm25_embedding(texts),
    ),
    log_level=logging.DEBUG
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What is FRP?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What is FRP?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What is FRP?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What is FRP?", param=QueryParam(mode="hybrid"))
)

import json
import logging
import random
import re
from tqdm import tqdm
import concurrent.futures
from datasets import load_dataset
import jsonlines

from lightrag import LightRAG, QueryParam
from lightrag.llm import bm25_embedding, openai_complete_if_cache
from lightrag.utils import MODEL, EmbeddingFunc, logger


TEMPLATE = """Question: {}
Choices: 
(A) {}
(B) {}
(C) {}
(D) {}

Please think step by step and output the final answer in the format: [[X]] (X is A, B, C, or D), and denote any related document in the response in the format: ((X)) (X is the number of the document)."""

DATA_DIR = r"/flash2/aml/zjliu24/datasets/gpqa_formatted"
TEMPERATURE = 0
MAX_TOKENS = 8192 # 4096, 8192, 20480
TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_cft_rag/gpqa_eval_data.jsonl"
SKIP_ROW = 0
BASE_URL = "http://localhost:7689/v1"
API_KEY = "token-abc123"
MAX_WORKERS = 32
DATA_BASE_FILE = r"/flash2/aml/zjliu24/datasets/OpenScholar-DataStore-V3/passages/raw_passages-0-of-16.jsonl"
DATA_BASE_CNT = 100000
RAG_TOP_K = 5


def extract_text_from_jsonl(file_path, k):
    texts = []
    with jsonlines.open(file_path) as reader:
        for i, obj in enumerate(reader):
            if i >= k:
                break
            texts.append(obj.get('text', ''))  # 假设每个记录都有'text'字段
    return texts

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

rag_client = LightRAG(
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
rag_client.insert(extract_text_from_jsonl(DATA_BASE_FILE, DATA_BASE_CNT))

dataset = load_dataset(DATA_DIR, "main")['train']

def extract_last_X(s):
    matches = re.findall(r'\[\[([abcdABCD])\]\]|\[\boxed{([abcdABCD])}\]|\[([abcdABCD])\]|\((([abcdABCD]))\)', s)
    
    results = [match[0] if match[0] else match[1] if match[1] else match[2] if match[2] else match[3] for match in matches]
    
    return results[-1].upper() if results else None


def process_eid(eid, question, choices, answer, ref_option):
    ref_choice = choices[ord(ref_option) - ord('A')]
    rand_choices = choices.copy()
    random.shuffle(rand_choices)
    rand_ref_option = chr(ord('A') + rand_choices.index(ref_choice))
    try:
        response = rag_client.query(TEMPLATE.format(question, *rand_choices), param=QueryParam(mode="naive", top_k=RAG_TOP_K))
        resp_option = extract_last_X(response)
    except:
        response = ""
        resp_option = None
    response = [(response, resp_option, rand_ref_option, resp_option == rand_ref_option if resp_option else False)]
    return {
        'question': question,
        'options': choices,
        'ref_option': ref_option,
        'answer': answer,
        'responses': response
    }

dataset = dataset[SKIP_ROW:]
question_list = dataset['Question']
options_list = dataset['options']
answer_list = dataset['answer']

with open(TARGET_FILE, "a") as f:
    item_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_eid, eid, question, choices, choices[answer], chr(ord('A') + answer)) for eid, (question, choices, answer) in enumerate(zip(question_list, options_list, answer_list))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            item_list.append(future.result())
    
    for item in item_list:
        f.write(json.dumps(item) + "\n")

# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json
import random
import re
from tqdm import tqdm
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset


TEMPLATE = """Question: {}
Choices: 
(A) {}
(B) {}
(C) {}
(D) {}

Please think step by step and output the final answer in the format: [[X]] (X is A, B, C, or D)."""
# MODEL = "/flash2/aml/public/models/Llama-3.1-8B-Instruct"
# MODEL = "/flash2/aml/zjliu24/gpqa_agent/post_train/training_rft"
MODEL = "/flash2/aml/zjliu24/gpqa_agent/post_train/training_cft"
# MODEL = "/flash2/aml/public/models/chatglm3-6b"
# MODEL = "/flash2/aml/zjliu24/gpqa_agent/post_train/training_inst_rft"
# MODEL = "/flash2/aml/zjliu24/gpqa_agent/post_train/training_inst_cft"
DATA_DIR = r"/flash2/aml/zjliu24/datasets/gpqa_formatted"
TEMPERATURE = 0
MAX_TOKENS = 20480 # 4096, 8192, 20480
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_llama3_1_8b_inst/gpqa_eval_data.jsonl"
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_rft/gpqa_eval_data.jsonl"
TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_cft/gpqa_eval_data.jsonl"
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_6b/gpqa_eval_data.jsonl"
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_llama3_1_rft/gpqa_eval_data.jsonl"
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_llama3_1_cft/gpqa_eval_data.jsonl"
SKIP_ROW = 0
MAX_WOKERS = 4 # 4, 32


client = OpenAI(
        base_url=f"http://localhost:7689/v1",
        api_key="token-abc123",
        timeout=300
    )

dataset = load_dataset(DATA_DIR, "main")['train']

def call_llm(question, choices):
    prompt = TEMPLATE.format(question, *choices)
    response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature = TEMPERATURE,
                max_tokens = MAX_TOKENS
            )
    return response.choices[0].message.content

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
        response = call_llm(question, rand_choices)
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WOKERS) as executor:
        futures = [executor.submit(process_eid, eid, question, choices, choices[answer], chr(ord('A') + answer)) for eid, (question, choices, answer) in enumerate(zip(question_list, options_list, answer_list))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            item_list.append(future.result())
    
    for item in item_list:
        f.write(json.dumps(item) + "\n")

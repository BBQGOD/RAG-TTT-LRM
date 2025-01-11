# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json
import random
import re
import concurrent.futures
from openai import OpenAI
from datasets import load_dataset

from template import TEMPLATE

MODEL = "/flash2/aml/public/models/QwQ-32B-Preview"
DATA_DIR = r"/flash2/aml/zjliu24/datasets/gpqa_formatted"
TEMPERATURE = 0.5
MAX_TOKENS = 20480
TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data.jsonl"
ANS_NUM = 64
SKIP_ROW = 0


client = OpenAI(
        base_url=f"http://localhost:7688/v1",
        api_key="token-abc123",
        timeout=600
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

def process_aid(aid, question, choices, ref_option):
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
    return (response, resp_option, rand_choices, rand_ref_option, resp_option == rand_ref_option if resp_option else False)

dataset = dataset[SKIP_ROW:]
question_list = dataset['Question']
options_list = dataset['options']
answer_list = dataset['answer']

with open(TARGET_FILE, "a") as f:
    for eid, example in enumerate(zip(question_list, options_list, answer_list)):
        print(f"Processing example {eid}")
        resp_list = []
        question = example[0]
        choices = example[1]
        answer = choices[example[2]]
        ref_option = chr(ord('A') + example[2])
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_aid, aid, question, choices, ref_option) for aid in range(ANS_NUM)]
            for future in concurrent.futures.as_completed(futures):
                resp_list.append(future.result())

        item = {
            'question': question,
            'options': choices,
            'ref_option': ref_option,
            'answer': answer,
            'responses': resp_list
        }
        f.write(json.dumps(item) + "\n")

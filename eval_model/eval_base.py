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
MODEL = "/flash2/aml/public/models/chatglm3-6b-base"
DATA_DIR = r"/flash2/aml/zjliu24/datasets/gpqa_formatted"
TEMPERATURE = 0
SHOT_NUM = 0
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_6b_base/gpqa_eval_data.jsonl"
# TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_6b_base_1s/gpqa_eval_data.jsonl"
TARGET_FILE = "/flash2/aml/zjliu24/h13_data/eval_model_chatglm3_6b_base_0s/gpqa_eval_data.jsonl"
SKIP_ROW = 0


client = OpenAI(
        base_url=f"http://localhost:7689/v1",
        api_key="token-abc123",
        timeout=300
    )

dataset = load_dataset(DATA_DIR, "main")['train']

dataset = dataset[SKIP_ROW:]
question_list = dataset['Question']
options_list = dataset['options']
answer_list = dataset['answer']

def process_few_shot(few_shot_list):
    few_shot_prompt = ""
    for question, choices, answer in few_shot_list:
        ref_choice = choices[answer]
        rand_choices = choices.copy()
        random.shuffle(rand_choices)
        rand_ref_option = chr(ord('A') + rand_choices.index(ref_choice))
        few_shot_prompt += TEMPLATE.format(question, *rand_choices) + "\n[[" + rand_ref_option + "]]\n\n"
    
    return few_shot_prompt.strip()

def call_llm(question, choices):
    few_shot_list = list(zip(question_list, options_list, answer_list))
    few_shot_list = random.sample(few_shot_list, SHOT_NUM)
    few_shot_prompt = process_few_shot(few_shot_list)

    prompt = few_shot_prompt + ("\n\n" if few_shot_prompt else "") + TEMPLATE.format(question, *choices)
    response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature = TEMPERATURE,
                extra_body={
                    'guided_choice': ['[[A]]', '[[B]]', '[[C]]', '[[D]]']
                }
            )
    return response.choices[0].message.content

def extract_last_X(s):
    matches = re.findall(r'\[\[([abcdABCD])\]\]|\[\boxed{([abcdABCD])}\]', s)
    results = [match[0] if match[0] else match[1] for match in matches]
    
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

with open(TARGET_FILE, "a") as f:
    item_list = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_eid, eid, question, choices, choices[answer], chr(ord('A') + answer)) for eid, (question, choices, answer) in enumerate(zip(question_list, options_list, answer_list))]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing"):
            item_list.append(future.result())
    
    for item in item_list:
        f.write(json.dumps(item) + "\n")

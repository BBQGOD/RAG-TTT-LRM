# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from template import CONF_TEMPLATE

MODEL = "/flash2/aml/public/models/QwQ-32B-Preview"
TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data_update.jsonl"
OUTPUT_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data_confidence.jsonl"
TOT_CONF_LEVEL = 10
CONF_LEVELS = ["((" + str(conf) + "))" for conf in range(0, TOT_CONF_LEVEL)]
TEMPERATURE = 0

client = OpenAI(
        base_url=f"http://localhost:7688/v1",
        api_key="token-abc123",
        timeout=600
    )

def call_llm(question, choices, response):
    prompt = CONF_TEMPLATE.format(question, *choices, response, TOT_CONF_LEVEL, TOT_CONF_LEVEL)

    response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature = TEMPERATURE,
                extra_body={
                    'guided_choice': CONF_LEVELS
                }
            )
    return response.choices[0].message.content

def convert_conf(conf_str):
    return CONF_LEVELS.index(conf_str) / TOT_CONF_LEVEL

# 使用并发计算处理每个响应项
def process_responses(question, resp_list):
    t_resp_list = []
    with ThreadPoolExecutor() as executor:
        future_to_resp = {executor.submit(call_llm, question, resp_item[2], resp_item[0]): resp_item for resp_item in resp_list}
        for future in as_completed(future_to_resp):
            resp_item = future_to_resp[future]
            try:
                conf_str = future.result()
                conf = convert_conf(conf_str)
                t_resp_list.append((resp_item[0], resp_item[1], resp_item[2], resp_item[3], resp_item[4], conf))
            except Exception as e:
                print(f"Error processing {resp_item[0]}: {e}")
    return t_resp_list

res_list = []
with open(TARGET_FILE, 'r') as f:
    for lid, line in enumerate(f):
        print(f"Processing Sample {lid}")
        item = json.loads(line)
        question = item['question']
        resp_list = item['responses']
        # 使用并发计算处理响应
        t_resp_list = process_responses(question, resp_list)
        item['responses'] = t_resp_list
        res_list.append(item)

print(f"Total {len(res_list)} Samples")

# 将结果写入输出文件
with open(OUTPUT_FILE, 'w') as f:
    for res_item in res_list:
        f.write(json.dumps(res_item, ensure_ascii=False) + '\n')

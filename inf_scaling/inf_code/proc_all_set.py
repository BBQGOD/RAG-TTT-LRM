# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json

from template import TEMPLATE

TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data_update.jsonl"
OUTPUT_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/training_set_all.json"
PER_SAMPLE_DATA_NUM = 12


sample_list = []
with open(TARGET_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        question = item['question']
        resp_list = item['responses']
        selected_responses = resp_list[:PER_SAMPLE_DATA_NUM]

        valid_responses = [
            {
                "conversations": [
                    {
                        "from": "human",
                        "value": TEMPLATE.format(question, *resp_item[2])
                    },
                    {
                        "from": "gpt",
                        "value": resp_item[0]
                    }
                ]
            }
            for resp_item in selected_responses
        ]
        sample_list.extend(valid_responses)

print(f"Total {len(sample_list)} Samples")

with open(OUTPUT_FILE, 'w') as f:
    f.write(json.dumps(sample_list, indent=4, ensure_ascii=False))

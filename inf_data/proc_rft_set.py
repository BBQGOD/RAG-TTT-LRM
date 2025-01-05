# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json

from template import TEMPLATE

TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data.jsonl"
OUTPUT_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/training_set_rft.json"


sample_list = []
skipped = 0
with open(TARGET_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        question = item['question']
        resp_list = item['responses']

        true_count = sum(1 for resp_item in resp_list if resp_item[-1])

        if true_count > 0:
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
                for resp_item in resp_list if resp_item[-1]
            ]
            sample_list.extend(valid_responses)
        else:
            skipped += 1
            print(f"Example {question} has less than half correct responses, skipping.")

print(f"Skipped {skipped} examples.") # 95

with open(OUTPUT_FILE, 'w') as f:
    f.write(json.dumps(sample_list, indent=4, ensure_ascii=False))
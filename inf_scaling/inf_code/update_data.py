# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json
import re

TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data.jsonl"
OUTPUT_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data_update.jsonl"


def extract_last_X(s):
    matches = re.findall(r'\[\[([abcdABCD])\]\]|\[\boxed{([abcdABCD])}\]|\[([abcdABCD])\]|\((([abcdABCD]))\)', s)
    
    results = [match[0] if match[0] else match[1] if match[1] else match[2] if match[2] else match[3] for match in matches]
    
    return results[-1].upper() if results else None

res_list = []
ori_none_cnt = 0
none_cnt = 0
with open(TARGET_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        resp_list = item['responses']
        t_resp_list = []
        for resp_item in resp_list:
            if resp_item[1] is None:
                ori_none_cnt += 1
            resp_option = extract_last_X(resp_item[0])
            if resp_option is None:
                none_cnt += 1
            t_resp_list.append((resp_item[0], resp_option, resp_item[2], resp_item[3], resp_option == resp_item[3] if resp_option else False))
        item['responses'] = t_resp_list
        res_list.append(item)

print(f"Total {len(res_list)} Samples")
print(f"Total {ori_none_cnt} Original None Values")
print(f"Total {none_cnt} None Values")

with open(OUTPUT_FILE, 'w') as f:
    for res_item in res_list:
        f.write(json.dumps(res_item, ensure_ascii=False) + '\n')

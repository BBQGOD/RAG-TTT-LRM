import json

from template import TEMPLATE

TARGET_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data.jsonl"
OUTPUT_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/training_set_cft.json"


sample_list = []
skipped = 0
corrected = 0
with open(TARGET_FILE, 'r') as f:
    for line in f:
        item = json.loads(line)
        question = item['question']
        resp_list = item['responses']
        resp_list = [resp_item for resp_item in resp_list if resp_item[1]]
        if len(resp_list) < 2:
            skipped += 1
            print(f"Example {question} has less than two responses, skipping.")
            continue
        choice_list = [resp_item[2][ord(resp_item[1]) - ord('A')] for resp_item in resp_list]
        maj_choice = max(set(choice_list), key=choice_list.count)

        selected_indices = [i for i, choice in enumerate(choice_list) if choice == maj_choice]
        selected_responses = [resp_list[i] for i in selected_indices]
        if selected_responses[0][-1]:
            corrected += 1

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

print(f"Skipped {skipped} examples.") # 66
print(f"Corrected {corrected} examples.") # 205
print(f"Total {len(sample_list)} Samples")

with open(OUTPUT_FILE, 'w') as f:
    f.write(json.dumps(sample_list, indent=4, ensure_ascii=False))
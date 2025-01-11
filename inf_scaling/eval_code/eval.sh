#!/bin/bash

python collect_data.py > /flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/eval_time.log
python eval_data.py > /flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/eval_result.log

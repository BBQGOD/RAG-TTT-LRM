# Retrieval-Augmented Generation (RAG) Meets Per-Domain Test-Time Training (TTT)

## Environment Setup

1. Download `jeggers/gpqa_formatted` dataset and `OpenScholar/OpenScholar-DataStore-V3` dataset from [huggingface.co](huggingface.co). And also download `Qwen/QwQ-32B-Preview`, `meta-llama/Llama-3.1-8B-Instruct`, `THUDM/chatglm3-6b`, and `THUDM/chatglm3-6b-base` models. Remember to replace all the paths in the code with your own paths.

2. Replace all the linked folders with your own folders.

3. Run the following command to install the required packages:

```bash
pip install vllm # all models should be served with vllm or OpenAI-API-compatible service
pip install -r rag/requirements.txt
```

4. Serve all models with vllm or OpenAI-API-compatible service, and change the `base_url` and `api_key` in each file to your own.

## Building RAG Knowledge Base

We use the first 100000 chunks in the `OpenScholar/OpenScholar-DataStore-V3` dataset to build the RAG knowledge base, which could be configured in `rag/eval_rag.py`.

## Collect Per-Domain TTT Data

TTT data are collected by majority voting with `Qwen/QwQ-32B-Preview`. We also provide groundtruth-based baseline training set.

```bash
cd inf_data
python collect_data.py
python proc_cft_set.py # generate majority voted data
python proc_rft_set.py # generate groundtruth-based baseline data
```

## Evaluation on GPQA Benchmark

Remember to configure the hyper-parameters in each file.

### Bare LLMs

```bash
cd eval_model
python eval_base.py # perform few-shot evaluation on pretrained models
python eval_instruct.py # perform zero-shot evaluation on instruct models
```

### LLMs with RAG

```bash
cd rag
python eval_rag.py
```

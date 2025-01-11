# Scaling Inference Compute of Long Reasoning Models with Small Proxies

This project implements a method for scaling inference compute of long reasoning LLMs with small proxy models, aiming to achieve high performance with constrained computational costs. 

## Methodology Overview

![method](./method.png)

Upon the figure, the left part (a) represents rolling out Long Reasoning Models (LRMs) to generate reasoning chains, and then generate self-confidence labels to perform weighted majority voting. The approach of weighted majority voting is to assign weights to each response based on the confidence labels, and then select the answer with the highest weighted count, whose formulation is as follows:
$$
\boxed{
E[\text{WeightedMajN}] = \frac{1}{\binom{M}{N}} \sum_{S \subseteq \{R_1, \dots, R_M\}, |S| = N} \arg \max_{A_i \in \{A_1, A_2, \dots, A_n\}} \sum_{R \in S} \mathbf{1}_{\{\hat{A}_{R} = A_i\}} \cdot \text{Conf}_R,
}
$$
where $ \text{Conf}_R $ is the confidence label of response $ R $. Specifically, the confidence labels can be mapped into any range $[0.5-\epsilon, 0.5+\epsilon]$ with length $2\epsilon$.

As a special case, `BestConf@N` selects the response with the highest confidence label, represented as:
$$
\boxed{
E[\text{BestConfN}] = \frac{1}{\binom{M}{N}} \sum_{S \subseteq \{R_1, \dots, R_M\}, |S| = N} \arg \max_{A_i \in \{A_1, A_2, \dots, A_n\}} \mathbf{1}_{\{\hat{A}_{R} = A_i\}} \cdot \text{Conf}_R.
}
$$

In the figure above, the right part (b) represents the method of using small proxies to generate reasoning chains, and then use the LRM to generate confidence labels, aiming to combine the effectiveness of LRMs and the efficiency of small proxies.

![perf-time-2](./perf_2.png)

From the trends shown in the empirical results above, it might be evident that smaller the proxy model is, better the performance improvement is.

> See the course report (`report.md`) for further details.

## Inference Compute Scaling

1. Perform sampling on LRMs or small proxies:

    ```bash
    cd proxy_inf_code
    python collect_data_*.py # Choose the script based on the model type and change the paths
    ```

2. Generate confidence labels:

    ```bash
    cd proxy_inf_code
    python inf_confidence.py # Change the paths
    ```

## Evaluation on GPQA Benchmark

```bash
cd eval_code
bash eval.sh # Change the paths
```

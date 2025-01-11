# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

import json
import math
import itertools
import numpy as np
from typing import List
from collections import Counter
import concurrent.futures

CUTOFF_RANGE = 0.1

def expected_PassN(M, T, N):
    """
    计算期望 PassN（至少有一条正确回复的概率）。

    参数:
    M (int): 总回复数。
    T (int): 正确回复数。
    N (int): 抽取的回复数。

    返回:
    float: 期望 PassN 的值。
    """
    if not (0 <= T <= M):
        raise ValueError("正确回复数 T 必须满足 0 ≤ T ≤ M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if M == 0:
        raise ValueError("总回复数 M 必须大于 0。")
    if N == 0:
        return 0.0  # 抽取0条回复，至少有一条正确的概率为0

    # 如果 T == 0，则所有回复都是错误的，PassN = 0
    if T == 0:
        return 0.0
    # 如果 T >=1 且 N > M - T，则至少有一条正确回复的概率为1
    if T >=1 and N > M - T:
        return 1.0

    try:
        total_comb = math.comb(M, N)
        wrong_comb = math.comb(M - T, N)
        probability = 1 - wrong_comb / total_comb
        return probability
    except ValueError:
        # 当 M - T < N 时，math.comb(M - T, N) 会抛出错误，此时至少有一条正确回复的概率为1
        return 1.0

# 复杂度：O(\binom{M}{N})
def expected_majn_exact(M: int, choice_list: List[str], N: int, correct_choice: str) -> float:
    """
    精确计算期望 MajN。
    """
    if len(choice_list) != M:
        raise ValueError("答案列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    majn_sum = 0.0
    total_subsets = 0
    
    for subset in itertools.combinations(range(M), N):
        sub_choice_list = [choice_list[i] for i in subset]
        
        counter = Counter(sub_choice_list)
        if None in counter:
            del counter[None]

        max_count = max(counter.values()) if counter else 0
        most_frequent_choices = [choice for choice, count in counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
        total_subsets += 1
    
    if total_subsets == 0:
        return 0.0
    
    expected_majn = majn_sum / total_subsets
    return expected_majn

# 复杂度：O(\binom{M}{N})
def expected_majn_weighted_exact(M: int, choice_list: List[str], N: int, correct_choice: str, weights: List[float]) -> float:
    """
    精确计算带权重的期望 MajN。
    """
    if len(choice_list) != M or len(weights) != M:
        raise ValueError("答案列表的长度和权重列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    majn_sum = 0.0
    total_subsets = 0
    
    for subset in itertools.combinations(range(M), N):
        sub_choice_list = [choice_list[i] for i in subset]
        
        weighted_counter = Counter()
        for i in range(N):
            if sub_choice_list[i] is not None:
                weighted_counter[sub_choice_list[i]] += weights[subset[i]]
        
        max_count = max(weighted_counter.values()) if weighted_counter else 0
        most_frequent_choices = [choice for choice, count in weighted_counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
        total_subsets += 1
    
    if total_subsets == 0:
        return 0.0
    
    expected_majn = majn_sum / total_subsets
    return expected_majn

# 复杂度：O(\binom{M}{N})
def expected_majn_weighted_cuttail_exact(M: int, choice_list: List[str], N: int, correct_choice: str, weights: List[float]) -> float:
    """
    精确计算带权重的期望 MajN。
    """
    if len(choice_list) != M or len(weights) != M:
        raise ValueError("答案列表的长度和权重列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    majn_sum = 0.0
    total_subsets = 0
    
    for subset in itertools.combinations(range(M), N):
        sub_choice_list = [choice_list[i] for i in subset]
        # sub_weights = [weights[i] for i in subset]
        # sub_weights = [0.3 if weights[i] < 0.3 else (0.6 if weights[i] > 0.6 else weights[i]) for i in subset]
        sub_weights = [weights[i] * CUTOFF_RANGE + (0.5 - CUTOFF_RANGE / 2) for i in subset]
        # avg_weight = sum(sub_weights) / N
        
        weighted_counter = Counter()
        for i in range(N):
            if sub_choice_list[i] is not None:
                weighted_counter[sub_choice_list[i]] += sub_weights[i]
        
        max_count = max(weighted_counter.values()) if weighted_counter else 0
        most_frequent_choices = [choice for choice, count in weighted_counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
        total_subsets += 1
    
    if total_subsets == 0:
        return 0.0
    
    expected_majn = majn_sum / total_subsets
    return expected_majn

# 复杂度：O(\binom{M}{N})
def expected_bon_exact(M: int, T: int, N: int, weights: List[float]) -> float:
    """
    精确计算期望 BoN。
    """
    if len(weights) != M:
        raise ValueError("权重列表的长度必须等于总回复数 M。")
    if not (0 <= T <= M):
        raise ValueError("正确回复数 T 必须满足 0 ≤ T ≤ M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    correct_indices = set(range(T))
    majn_sum = 0.0
    total_subsets = 0
    
    for subset in itertools.combinations(range(M), N):
        max_w = max(weights[i] for i in subset)
        max_indices = [i for i in subset if weights[i] == max_w]
        max_correctness = [i in correct_indices for i in max_indices]
        
        contrib = sum(max_correctness) / len(max_correctness)
        majn_sum += contrib
        total_subsets += 1
    
    if total_subsets == 0:
        return 0.0
    
    expected_majn = majn_sum / total_subsets
    return expected_majn

def expected_majn_monte_carlo(M: int, choice_list: List[str], N: int, correct_choice: str, num_simulations: int = 100000) -> float:
    """
    蒙特卡洛模拟计算期望 MajN。
    """
    if len(choice_list) != M:
        raise ValueError("答案列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    choice_array = np.array(choice_list)
    majn_sum = 0.0
    
    for _ in range(num_simulations):
        subset_indices = np.random.choice(M, size=N, replace=False)
        sub_choice_list = choice_array[subset_indices]

        counter = Counter(sub_choice_list)
        if None in counter:
            del counter[None]
        max_count = max(counter.values()) if counter else 0
        most_frequent_choices = [choice for choice, count in counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
    
    expected_majn = majn_sum / num_simulations
    return expected_majn

def expected_majn_weighted_monte_carlo(M: int, choice_list: List[str], N: int, correct_choice: str, weights: List[float], num_simulations: int = 100000) -> float:
    """
    蒙特卡洛模拟计算带权重的期望 MajN。
    """
    if len(choice_list) != M or len(weights) != M:
        raise ValueError("答案列表的长度和权重列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    choice_array = np.array(choice_list)
    weights_array = np.array(weights)
    majn_sum = 0.0
    
    for _ in range(num_simulations):
        subset_indices = np.random.choice(M, size=N, replace=False)
        sub_choice_list = choice_array[subset_indices]
        
        weighted_counter = Counter()
        for i in range(N):
            if sub_choice_list[i] is not None:
                weighted_counter[sub_choice_list[i]] += weights_array[subset_indices[i]]
        
        max_count = max(weighted_counter.values()) if weighted_counter else 0
        most_frequent_choices = [choice for choice, count in weighted_counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
    
    expected_majn = majn_sum / num_simulations
    return expected_majn

def expected_majn_weighted_cuttail_monte_carlo(M: int, choice_list: List[str], N: int, correct_choice: str, weights: List[float], num_simulations: int = 100000) -> float:
    """
    蒙特卡洛模拟计算带权重的期望 MajN。
    """
    if len(choice_list) != M or len(weights) != M:
        raise ValueError("答案列表的长度和权重列表的长度必须等于总回复数 M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    choice_array = np.array(choice_list)
    weights_array = np.array(weights)
    majn_sum = 0.0
    
    for _ in range(num_simulations):
        subset_indices = np.random.choice(M, size=N, replace=False)
        sub_choice_list = choice_array[subset_indices]
        # sub_weights = weights_array[subset_indices]
        # sub_weights = np.clip(weights_array[subset_indices], 0.3, 0.6)
        sub_weights = weights_array[subset_indices] * CUTOFF_RANGE + (0.5 - CUTOFF_RANGE / 2)
        # avg_weight = sub_weights.mean()
        
        weighted_counter = Counter()
        for i in range(N):
            if sub_choice_list[i] is not None:
                weighted_counter[sub_choice_list[i]] += sub_weights[i]
        
        max_count = max(weighted_counter.values()) if weighted_counter else 0
        most_frequent_choices = [choice for choice, count in weighted_counter.items() if count == max_count]
        
        if correct_choice in most_frequent_choices:
            if len(most_frequent_choices) == 1:
                contrib = 1.0
            else:
                contrib = 1.0 / len(most_frequent_choices)
        else:
            contrib = 0.0
        
        majn_sum += contrib
    
    expected_majn = majn_sum / num_simulations
    return expected_majn

def expected_bon_monte_carlo(M: int, T: int, N: int, weights: List[float], num_simulations: int = 100000) -> float:
    """
    蒙特卡洛模拟计算期望 BoN。
    """
    if len(weights) != M:
        raise ValueError("权重列表的长度必须等于总回复数 M。")
    if not (0 <= T <= M):
        raise ValueError("正确回复数 T 必须满足 0 ≤ T ≤ M。")
    if not (0 <= N <= M):
        raise ValueError("抽取数 N 必须满足 0 ≤ N ≤ M。")
    if N == 0:
        return 0.0
    
    weights_array = np.array(weights)
    correct_indices = set(range(T))
    majn_sum = 0.0
    
    for _ in range(num_simulations):
        subset = np.random.choice(M, size=N, replace=False)
        max_w = weights_array[subset].max()
        max_indices = subset[weights_array[subset] == max_w]
        max_correctness = [i in correct_indices for i in max_indices]
        
        contrib = sum(max_correctness) / len(max_correctness)
        
        majn_sum += contrib
    
    expected_majn = majn_sum / num_simulations
    return expected_majn

if __name__ == "__main__":
    # SRC_FILE = "/flash2/aml/zjliu24/h20_data/inf_data_qwq_preview/gpqa_inf_data_confidence.jsonl"
    # SRC_FILE = "/flash2/aml/zjliu24/h20_data/inf_conf_data/llama3_1_inst_gpqa_inf_data_confidence.jsonl"
    SRC_FILE = "/flash2/aml/zjliu24/h20_data/inf_conf_data/chatglm3_gpqa_inf_data_confidence.jsonl"
    N_RANGE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    MONTE_CORLO_THRESHOLD = 10**3

    # 定义用于处理每行数据的函数
    def process_line(line, n):
        item = json.loads(line)
        resp_list = item['responses']
        m = len(resp_list)
        if n > m:
            return None  # 若n > m，则跳过当前行

        t = len([resp_item for resp_item in resp_list if resp_item[4]])
        correct_choice = item['answer']
        choice_list = [resp_item[2][ord(resp_item[1]) - ord('A')] if resp_item[1] else None for resp_item in resp_list]
        conf_list = [resp_item[-1] for resp_item in resp_list]
        t_conf_list = [resp_item[-1] for resp_item in resp_list if resp_item[4]] + [resp_item[-1] for resp_item in resp_list if not resp_item[4]]

        pass_n = expected_PassN(m, t, n)

        if math.comb(m, n) <= MONTE_CORLO_THRESHOLD:
            maj_n = expected_majn_exact(m, choice_list, n, correct_choice)
            weighted_maj_n = expected_majn_weighted_exact(m, choice_list, n, correct_choice, conf_list)
            bon = expected_bon_exact(m, t, n, t_conf_list)

            weighted_maj_n_cuttail = expected_majn_weighted_cuttail_exact(m, choice_list, n, correct_choice, conf_list)
        else:
            maj_n = expected_majn_monte_carlo(m, choice_list, n, correct_choice, MONTE_CORLO_THRESHOLD)
            weighted_maj_n = expected_majn_weighted_monte_carlo(m, choice_list, n, correct_choice, conf_list, MONTE_CORLO_THRESHOLD)
            bon = expected_bon_monte_carlo(m, t, n, t_conf_list, MONTE_CORLO_THRESHOLD)

            weighted_maj_n_cuttail = expected_majn_weighted_cuttail_monte_carlo(m, choice_list, n, correct_choice, conf_list, MONTE_CORLO_THRESHOLD)

        return (pass_n, maj_n, weighted_maj_n, bon, weighted_maj_n_cuttail)

    for n in N_RANGE:
        pass_n_list = []
        maj_n_list = []
        weighted_maj_n_list = []
        bon_list = []
        weighted_maj_n_cuttail_list = []
        
        # 使用 ThreadPoolExecutor 来并行处理每行数据
        with open(SRC_FILE, 'r') as f:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                # 提交每一行的任务
                for line in f:
                    futures.append(executor.submit(process_line, line, n))
                
                # 获取每个任务的结果
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        pass_n, maj_n, weighted_maj_n, bon, weighted_maj_n_cuttail = result
                        pass_n_list.append(pass_n)
                        maj_n_list.append(maj_n)
                        weighted_maj_n_list.append(weighted_maj_n)
                        bon_list.append(bon)
                        weighted_maj_n_cuttail_list.append(weighted_maj_n_cuttail)

        pass_n_mean = np.mean(pass_n_list)
        maj_n_mean = np.mean(maj_n_list)
        weighted_maj_n_mean = np.mean(weighted_maj_n_list)
        bon_mean = np.mean(bon_list)
        weighted_maj_n_cuttail_mean = np.mean(weighted_maj_n_cuttail_list)
        print(f"N = {n}: PassN = {pass_n_mean:.4f}, MajN = {maj_n_mean:.4f}, Weighted MajN = {weighted_maj_n_mean:.4f}, Best-of-N = {bon_mean:.4f}, Weighted MajN Cuttail = {weighted_maj_n_cuttail_mean:.4f}", flush=True)

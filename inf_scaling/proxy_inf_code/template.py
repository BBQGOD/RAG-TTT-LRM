# Copyright (C) 2025  Zijun Liu, AML Course in 2024 Fall

TEMPLATE = """Question: {}
Choices: 
(A) {}
(B) {}
(C) {}
(D) {}

Please think step by step and output the final answer in the format: [[X]] (X is A, B, C, or D)."""

CONF_TEMPLATE = """Question: {}
Choices: 
(A) {}
(B) {}
(C) {}
(D) {}

Please think step by step and output the final answer in the format: [[X]] (X is A, B, C, or D).

Response: {}

Please rate your confidence level on the correctness of your answer in the range of 0 to {} in the format: ((N)) (N is an integer from 0 to {})."""

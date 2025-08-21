# Main metrics as used by WikiDYK: https://github.com/zhang-yu-wei/WikiDYK/blob/main/src/utils/metrics.py#L89

import re
import string
import numpy as np
from collections import Counter
from typing import Dict, Union

def normalize_answer(s: str) -> str:
    """
    Normalize text by lowercasing, removing punctuation, 
    removing articles (a, an, the), and fixing whitespace.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def qa_f1_score(prediction: str, ground_truth: str, model: str = "claude-3-opus-20240229") -> float:
    """
    Compute token-level F1 between a prediction and ground truth 
    after normalization and whitespace tokenization.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1(preds: str, labels: str) -> float:
    """
    Compute F1 score (%) for a prediction and label pair.
    """
    return float(100.0 * qa_f1_score(preds, labels))


def match(target: str, output: str) -> bool:
    """
    Return True if un-normalized target is a substring of un-normalized output.
    True/false questions are the exception: here, case-insensitive matiching is reported.
    """
    target_bool = target.lower()
    if target_bool in ["true", "false"]:
        return target_bool in output.lower()

    return target in output


def metrics(output: str, correct_answer: str) -> Dict[str, Union[bool, float]]:
    """
    Compute both Match and F1 metrics for a single output–answer pair.
    
    Args:
        output: The model's output string
        correct_answer: The expected correct answer string
        
    Returns:
        Dict with "match" (bool) and "f1" (float) scores
    """
    metric_result = {}
    metric_result["match"] = match(correct_answer, output)
    metric_result["f1"] = f1(output, correct_answer)
    return metric_result

# Example
# print(metrics("The Eiffel Tower is in Paris.", "Eiffel Tower is in"))
# {"match": True, "f1": 100.0}
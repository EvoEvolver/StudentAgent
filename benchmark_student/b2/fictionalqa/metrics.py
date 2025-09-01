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


def match(target: str, output: str) -> bool:
    """
    Return True if is a substring of the output (case insensitive).
    """
    return target.lower() in output.lower()


def metrics(output: str, correct_answer: str) -> Dict[str, Union[bool, float]]:
    """
    Compute both Match and F1 metrics for a single output–answer pair.
    
    Args:
        output: The model's output string
        correct_answer: The expected correct answer string
        
    Returns:
        Dict with "match" (bool)
    """
    metric_result = {}
    metric_result["match"] = match(correct_answer, output)
    return metric_result

# Example
# print(metrics("The Eiffel Tower is in Paris.", "Eiffel Tower is in"))
# {"match": True, "f1": 100.0}
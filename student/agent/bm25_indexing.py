import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import string


# Download necessary NLTK resources
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


# Initialize stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def tokenize(text):
    """Tokenizes and preprocesses text."""
    tokens = word_tokenize(text.lower())  # Lowercasing and tokenization
    if len(tokens) > 1:
        tokens = [word for word in tokens if word not in stop_words and word not in punctuations]  # Remove stopwords & punctuation
    return tokens

def get_bm25_score(embed_src: list[str], stimuli_list: list[str]):
    """Calculates BM25 score between embed_src and stimuli_list."""
    # Tokenize the stimuli
    tokenized_stimuli = [tokenize(stimuli) for stimuli in stimuli_list]
    # Tokenize the embed_src
    tokenized_embed_src = [tokenize(src) for src in embed_src]
    
    scores = []
    try:
        bm25 = BM25Okapi(tokenized_embed_src)
        for stimuli in tokenized_stimuli:
            score = bm25.get_scores(stimuli)
            scores.append(max(score))
        
    except ZeroDivisionError as e:
        # edge case, if both lists are identical and len=1
        return [1]
    
    return scores

if __name__ == '__main__':
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown dog jumps over a sleepy fox!",
        "The dog is quick and brown.",
        "Foxes are generally quick and agile."
    ]

    query = "The quick brown fox jumps over the lazy dog."
    scores = get_bm25_score(documents, [query])
    print(scores)
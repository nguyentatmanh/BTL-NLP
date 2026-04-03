from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple
import time
import re


class TfidfRetriever:
    """
    Upgraded to BM25 Retriever!
    We keep the class name `TfidfRetriever` so we don't break main.py.
    BM25 provides significantly better recall for question answering term-matching.
    """

    def __init__(self):
        self.bm25_model = None
        self.contexts: List[str] = []
        self._is_built = False

    def _tokenize(self, text: str) -> List[str]:
        """Basic whitespace and punctuation tokenization."""
        text = text.lower()
        # Remove punctuation, split by whitespace
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build_index(self, contexts: List[str]):
        """Build BM25 index from a list of context strings."""
        start = time.perf_counter()
        
        # Tokenize all contexts for BM25
        print("  Tokenizing contexts for BM25...")
        tokenized_corpus = [self._tokenize(ctx) for ctx in contexts]
        
        print("  Building BM25 index...")
        self.bm25_model = BM25Okapi(tokenized_corpus)
        
        self.contexts = contexts
        self._is_built = True
        
        elapsed = time.perf_counter() - start
        print(f"  BM25 index built: {len(contexts)} contexts in {elapsed:.2f}s")

    @property
    def is_built(self) -> bool:
        return self._is_built

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Search for top-k most relevant contexts for the given query using BM25.
        Returns list of (index, normalized_score, context_text).
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        tokenized_query = self._tokenize(query)
        
        # Get raw BM25 scores
        doc_scores = self.bm25_model.get_scores(tokenized_query)
        
        # Normalize BM25 scores to 0-1 range so it plays nicely with reader_score
        # Basic min-max scaling mapped to 0-1, or just divide by max if max > 0
        max_score = np.max(doc_scores)
        if max_score > 0:
            normalized_scores = doc_scores / max_score
        else:
            normalized_scores = doc_scores

        # Get top-k indices sorted by score descending
        # np.argsort returns ascending, so we slice [::-1]
        top_indices = np.argsort(normalized_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((int(idx), float(normalized_scores[idx]), self.contexts[idx]))

        return results

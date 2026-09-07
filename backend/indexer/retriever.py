import math
import re
from typing import List
from pydantic import BaseModel
from backend.indexer.models import CodeChunk


class SearchResult(BaseModel):
    """
    Represents a merged retrieval search result.
    """
    chunk: CodeChunk
    score: float
    retrieval_type: str  # 'sparse', 'dense', or 'hybrid'


def tokenize_code(text: str) -> List[str]:
    """
    Tokenizes code into individual alphanumeric terms.
    Splits camelCase and snake_case, filters special chars, and lowercases.
    """
    if not text:
        return []
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text)
    # Replace underscores with spaces
    text = text.replace("_", " ")
    # Split camelCase by inserting spaces before uppercase letters
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    # Split and lowercase
    return [t.lower() for t in text.split() if t]


def tokenize_symbol_name(symbol_name: str) -> List[str]:
    """
    Tokenizes a symbol name, keeping both the exact representation and split parts.
    """
    if not symbol_name:
        return []
    tokens = [symbol_name.lower()]
    tokens.extend(tokenize_code(symbol_name))
    return tokens


class SimpleBM25Index:
    """
    A lightweight, deterministic in-memory BM25 index for Python code chunks.
    Weights symbol names higher than standard code content tokens.
    """

    def __init__(
        self,
        chunks: List[CodeChunk],
        symbol_weight: float = 5.0,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.chunks = chunks
        self.symbol_weight = symbol_weight
        self.k1 = k1
        self.b = b

        self.N = len(chunks)
        self.doc_token_counts = []
        self.doc_lengths = []
        self.df = {}
        self.idf = {}
        self.avgdl = 0.0

        if self.N > 0:
            self._fit()

    def _fit(self):
        total_length = 0
        for chunk in self.chunks:
            # Weight symbol name tokens higher if present
            symbol_tokens = tokenize_symbol_name(chunk.symbol_name) if chunk.symbol_name else []
            content_tokens = tokenize_code(chunk.content)

            # Build term frequency mapping for this chunk
            tf_map = {}
            for token in content_tokens:
                tf_map[token] = tf_map.get(token, 0.0) + 1.0

            for token in symbol_tokens:
                tf_map[token] = tf_map.get(token, 0.0) + self.symbol_weight

            self.doc_token_counts.append(tf_map)

            # Unweighted/total token count length
            doc_len = len(content_tokens) + len(symbol_tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # Record document frequency for each term
            for term in tf_map.keys():
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_length / self.N

        # Compute IDF for all terms
        for term, df_val in self.df.items():
            self.idf[term] = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

    def search(self, query: str, top_n: int = 10) -> List[tuple[CodeChunk, float]]:
        """
        Rank the chunks against the query using the BM25 formula.
        Returns a sorted list of (chunk, score) tuples.
        """
        if self.N == 0:
            return []

        query_tokens = tokenize_code(query)
        # Also treat the query as a potential symbol name search
        query_symbol_tokens = tokenize_symbol_name(query)
        search_terms = list(set(query_tokens + query_symbol_tokens))

        results = []
        for i in range(self.N):
            score = 0.0
            tf_map = self.doc_token_counts[i]
            doc_len = self.doc_lengths[i]

            for term in search_terms:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf_val = self.idf.get(term, 0.0)

                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                score += idf_val * (numerator / denominator)

            if score > 0.0:
                results.append((self.chunks[i], score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]


def _chunk_key(chunk: CodeChunk) -> str:
    """
    Generates a unique deduplication key for a CodeChunk.
    """
    return f"{chunk.file_path}:{chunk.symbol_name or ''}:{chunk.start_line}:{chunk.end_line}"


def reciprocal_rank_fusion(
    sparse_candidates: List[CodeChunk],
    dense_candidates: List[CodeChunk],
    k: int = 60,
) -> List[SearchResult]:
    """
    Performs Reciprocal Rank Fusion on sparse and dense retrieval lists.
    """
    rrf_scores = {}
    chunk_map = {}
    origins = {}

    # Rank sparse candidates
    for rank_idx, chunk in enumerate(sparse_candidates):
        key = _chunk_key(chunk)
        chunk_map[key] = chunk
        rank = rank_idx + 1
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        origins[key] = {"sparse"}

    # Rank dense candidates
    for rank_idx, chunk in enumerate(dense_candidates):
        key = _chunk_key(chunk)
        chunk_map[key] = chunk
        rank = rank_idx + 1
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))
        if key not in origins:
            origins[key] = set()
        origins[key].add("dense")

    # Form SearchResults
    results = []
    for key, score in rrf_scores.items():
        chunk_origins = origins[key]
        if "sparse" in chunk_origins and "dense" in chunk_origins:
            retrieval_type = "hybrid"
        elif "sparse" in chunk_origins:
            retrieval_type = "sparse"
        else:
            retrieval_type = "dense"

        results.append(
            SearchResult(
                chunk=chunk_map[key],
                score=score,
                retrieval_type=retrieval_type,
            )
        )

    results.sort(key=lambda x: x.score, reverse=True)
    return results


class HybridRetriever:
    """
    Hybrid retriever merging BM25 and vector results via Reciprocal Rank Fusion.
    """

    def retrieve(
        self,
        sparse_candidates: List[CodeChunk],
        dense_candidates: List[CodeChunk],
        top_k: int = 4,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Merge candidate lists from BM25 (sparse) and vector (dense) sources using RRF.
        """
        return reciprocal_rank_fusion(sparse_candidates, dense_candidates, k=k)[:top_k]

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from agent_memory.core.schema import Chunk, RetrievedChunk


def retrieve_top_k(chunks: list[Chunk], embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    if not chunks or embeddings.size == 0:
        return []
    scores = embeddings @ query_embedding.reshape(-1)
    order = np.argsort(scores)[::-1][:top_k]
    retrieved: list[RetrievedChunk] = []
    for rank, index in enumerate(order, start=1):
        chunk = chunks[int(index)]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=float(scores[int(index)]),
                rank=rank,
            )
        )
    return retrieved


def retrieve_lexical_top_k(
    chunks: list[Chunk],
    query: str,
    top_k: int,
    *,
    include_date: bool = False,
) -> list[RetrievedChunk]:
    if not chunks:
        return []

    query_terms = tokenize(query)
    if not query_terms:
        return []

    documents = [tokenize(lexical_document(chunk, include_date=include_date)) for chunk in chunks]
    term_counts = [Counter(document) for document in documents]
    doc_freq: Counter[str] = Counter()
    for document in documents:
        doc_freq.update(set(document))

    num_docs = len(chunks)
    avg_len = sum(len(document) for document in documents) / max(num_docs, 1)
    scores = []
    for index, counts in enumerate(term_counts):
        doc_len = len(documents[index])
        score = bm25_score(query_terms, counts, doc_freq, doc_len, avg_len, num_docs)
        if score > 0:
            scores.append((score, index))

    scores.sort(reverse=True)
    retrieved: list[RetrievedChunk] = []
    for rank, (score, index) in enumerate(scores[:top_k], start=1):
        chunk = chunks[index]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=float(score),
                rank=rank,
            )
        )
    return retrieved


def merge_ranked_results(result_sets: list[list[RetrievedChunk]], top_k: int, rrf_k: int = 60) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for results in result_sets:
        for result in results:
            chunks[result.chunk_id] = result
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + result.rank)

    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    merged: list[RetrievedChunk] = []
    for rank, chunk_id in enumerate(ordered, start=1):
        result = chunks[chunk_id]
        merged.append(
            RetrievedChunk(
                chunk_id=result.chunk_id,
                text=result.text,
                date=result.date,
                session_id=result.session_id,
                score=float(scores[chunk_id]),
                rank=rank,
            )
        )
    return merged


def bm25_score(
    query_terms: list[str],
    counts: Counter[str],
    doc_freq: Counter[str],
    doc_len: int,
    avg_len: float,
    num_docs: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    score = 0.0
    for term in query_terms:
        freq = counts.get(term, 0)
        if not freq:
            continue
        idf = math.log(1.0 + (num_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        denom = freq + k1 * (1.0 - b + b * doc_len / max(avg_len, 1e-9))
        score += idf * (freq * (k1 + 1.0)) / denom
    return score


def tokenize(text: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "current",
        "did",
        "different",
        "do",
        "for",
        "from",
        "have",
        "how",
        "i",
        "in",
        "is",
        "it",
        "many",
        "me",
        "much",
        "my",
        "number",
        "of",
        "on",
        "or",
        "the",
        "to",
        "total",
        "was",
        "were",
        "what",
        "when",
    }
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in stopwords]


def lexical_document(chunk: Chunk, *, include_date: bool) -> str:
    if include_date and chunk.date:
        return f"Date: {chunk.date}\nContent: {chunk.text}"
    return chunk.text

from __future__ import annotations

import re
from typing import Any

from agent_memory.baseline.queries import rerank_retrieved
from agent_memory.baseline.routing import (
    QuestionStrategy,
    RouteSettings,
    use_list_context_window_question,
    use_list_context_window_question_v2,
)
from agent_memory.core.schema import Chunk, Example, RetrievedChunk


def retrieval_search_space(
    chunks: list[Chunk],
    embeddings: Any,
    strategy: QuestionStrategy,
    retrieval_cfg: dict[str, Any],
) -> tuple[list[Chunk], Any]:
    if not bool(retrieval_cfg.get("turn_pair_chunks", False)):
        return chunks, embeddings
    allowed = {str(value) for value in retrieval_cfg.get("turn_pair_strategies", [])}
    if not allowed or strategy.name in allowed:
        return chunks, embeddings

    indices = [index for index, chunk in enumerate(chunks) if not is_turn_pair_chunk_id(chunk.chunk_id)]
    if len(indices) == len(chunks):
        return chunks, embeddings
    return [chunks[index] for index in indices], embeddings[indices]


def materialize_turn_pair_retrieval(
    chunks: list[Chunk],
    retrieved: list[RetrievedChunk],
    retrieval_cfg: dict[str, Any],
) -> list[RetrievedChunk]:
    if str(retrieval_cfg.get("turn_pair_answer_mode", "direct")) != "source_turns":
        return retrieved
    if not any(is_turn_pair_chunk_id(item.chunk_id) for item in retrieved):
        return retrieved

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    materialized: list[RetrievedChunk] = []
    seen: set[str] = set()

    def add(chunk: Chunk, source: RetrievedChunk, offset: int) -> None:
        if chunk.chunk_id in seen:
            return
        seen.add(chunk.chunk_id)
        materialized.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=source.score - offset * 1e-6,
                rank=len(materialized) + 1,
            )
        )

    for item in retrieved:
        pair_indices = turn_pair_indices(item.chunk_id)
        if pair_indices is None:
            add(
                Chunk(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    date=item.date,
                    session_id=item.session_id,
                ),
                item,
                0,
            )
            continue
        for offset, turn_index in enumerate(pair_indices):
            source_chunk = by_id.get(f"turn-{turn_index:05d}")
            if source_chunk is not None:
                add(source_chunk, item, offset)
    return rerank_retrieved(materialized)


def turn_pair_indices(chunk_id: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"turn-pair-(\d{5})-(\d{5})", str(chunk_id))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def is_turn_pair_chunk_id(chunk_id: str) -> bool:
    return str(chunk_id).startswith("turn-pair-")


def effective_context_window(
    settings: RouteSettings,
    strategy: QuestionStrategy,
    question: str,
    config: dict[str, Any],
) -> int:
    if settings.context_window <= 0:
        return 0
    question_mode = str((config.get("retrieval") or {}).get("context_window_question_mode", ""))
    if question_mode == "list_context_v1" and use_list_context_window_question(question):
        return settings.context_window
    if question_mode == "list_context_v2" and use_list_context_window_question_v2(question):
        return settings.context_window
    if settings.context_window_strategies and strategy.name not in settings.context_window_strategies:
        return 0
    return settings.context_window


def effective_context_window_top_n(
    settings: RouteSettings,
    strategy: QuestionStrategy,
    config: dict[str, Any],
) -> int:
    overrides = (config.get("retrieval") or {}).get("context_window_top_n_by_strategy") or {}
    if isinstance(overrides, dict) and strategy.name in overrides:
        return int(overrides[strategy.name])
    return settings.context_window_top_n


def maybe_prioritize_sum_financial_context(
    config: dict[str, Any],
    example: Example,
    retrieved: list[RetrievedChunk],
    route_plan: dict[str, Any] | None,
) -> list[RetrievedChunk]:
    answer_cfg = config.get("answer") or {}
    if not bool(answer_cfg.get("sum_financial_context_prioritize", False)):
        return retrieved
    if not route_plan or str(route_plan.get("operation") or "") != "sum":
        return retrieved
    if not is_financial_sum_question(example.question):
        return retrieved
    question_terms = significant_question_terms(example.question)
    if not question_terms:
        return retrieved

    def priority(item: RetrievedChunk, index: int) -> tuple[int, int, int]:
        text = item.text.lower()
        has_amount = bool(re.search(r"(?:\$|usd\b|dollars?\b|cents?\b)", text))
        overlap = sum(1 for term in question_terms if term in text)
        promoted = has_amount and overlap > 0
        return (0 if promoted else 1, -overlap, index)

    reordered = sorted(enumerate(retrieved), key=lambda pair: priority(pair[1], pair[0]))
    if [item.chunk_id for _, item in reordered] == [item.chunk_id for item in retrieved]:
        return retrieved
    return rerank_retrieved([item for _, item in reordered])


def is_financial_sum_question(question: str) -> bool:
    lowered = question.lower()
    return any(
        signal in lowered
        for signal in (
            "$",
            "money",
            "spent",
            "spend",
            "cost",
            "costs",
            "price",
            "paid",
            "raise",
            "raised",
            "fundraising",
            "expense",
            "expenses",
            "dollar",
            "dollars",
        )
    )


def significant_question_terms(question: str) -> set[str]:
    stopwords = {
        "about",
        "after",
        "before",
        "current",
        "from",
        "have",
        "how",
        "many",
        "much",
        "since",
        "that",
        "the",
        "this",
        "total",
        "what",
        "when",
        "where",
        "which",
        "with",
        "year",
        "years",
    }
    terms = set()
    for term in re.findall(r"[a-z0-9]+", question.lower()):
        if len(term) < 3 or term in stopwords:
            continue
        terms.add(term)
    return terms


def expand_retrieved_context(
    chunks: list[Chunk],
    retrieved: list[RetrievedChunk],
    *,
    window: int,
    top_n: int,
) -> list[RetrievedChunk]:
    if window <= 0 or not retrieved:
        return retrieved

    by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    limit = min(top_n if top_n > 0 else len(retrieved), len(retrieved))
    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()

    def append_chunk(chunk: Chunk, source: RetrievedChunk, offset: int) -> None:
        if chunk.chunk_id in seen:
            return
        seen.add(chunk.chunk_id)
        expanded.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=source.score - abs(offset) * 1e-6,
                rank=len(expanded) + 1,
            )
        )

    for source in retrieved[:limit]:
        index = by_id.get(source.chunk_id)
        if index is None:
            append_chunk(
                Chunk(
                    chunk_id=source.chunk_id,
                    text=source.text,
                    date=source.date,
                    session_id=source.session_id,
                ),
                source,
                0,
            )
            continue
        start = max(0, index - window)
        end = min(len(chunks), index + window + 1)
        for neighbor_index in range(start, end):
            neighbor = chunks[neighbor_index]
            if context_neighbor_allowed(source.chunk_id, neighbor.chunk_id):
                append_chunk(neighbor, source, neighbor_index - index)

    for source in retrieved[limit:]:
        if source.chunk_id in seen:
            continue
        expanded.append(
            RetrievedChunk(
                chunk_id=source.chunk_id,
                text=source.text,
                date=source.date,
                session_id=source.session_id,
                score=source.score,
                rank=len(expanded) + 1,
            )
        )
    return expanded


def context_neighbor_allowed(source_chunk_id: str, neighbor_chunk_id: str) -> bool:
    if is_turn_pair_chunk_id(source_chunk_id):
        return is_turn_pair_chunk_id(neighbor_chunk_id)
    if source_chunk_id.startswith("turn-"):
        return neighbor_chunk_id.startswith("turn-") and not is_turn_pair_chunk_id(neighbor_chunk_id)
    return True

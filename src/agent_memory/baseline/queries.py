from __future__ import annotations

from typing import Any

from agent_memory.core.schema import Example, RetrievedChunk
from agent_memory.baseline.routing import QuestionStrategy, asks_assistant_memory


def embedding_text(chunk: Any) -> str:
    if chunk.date:
        return f"Date: {chunk.date}\nContent: {chunk.text}"
    return chunk.text


def retrieval_query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def expanded_retrieval_query_text(example: Example) -> str:
    question = retrieval_query_text(example)
    return (
        f"{question}\n\n"
        "Find memories that provide direct evidence for this question, including relevant facts, dates, "
        "numbers, events, names, user preferences, dislikes, constraints, owned resources, prior experiences, "
        "habits, and goals."
    )


def expanded_lexical_query_text(example: Example) -> str:
    question = retrieval_query_text(example)
    terms = lexical_expansion_terms(example.question)
    if not terms:
        return question
    return f"{question}\nRelated terms: {' '.join(terms)}"


def lexical_expansion_terms(question: str) -> list[str]:
    lowered = question.lower()
    groups = [
        (
            ("acquire", "acquired", "buy", "bought", "purchase", "purchased", "order", "ordered"),
            ("got", "received", "picked up", "owned", "store", "shop"),
        ),
        (
            ("attend", "attended", "visit", "visited", "viewed", "complete", "completed"),
            ("went", "saw", "finished", "joined", "participated"),
        ),
        (
            ("work", "job", "role", "position", "career", "company", "occupation", "profession"),
            (
                "previous role",
                "new role",
                "title",
                "promoted",
                "senior",
                "specialist",
                "analyst",
                "coordinator",
                "team",
                "project",
                "startup",
                "responsible",
            ),
        ),
        (
            ("family", "sibling", "siblings", "relative"),
            ("brother", "brothers", "sister", "sisters", "parent", "parents"),
        ),
        (
            ("health", "doctor", "device", "devices", "medical"),
            ("fitbit", "watch", "meter", "monitor", "spray", "physician", "medication"),
        ),
        (
            ("education", "school", "university", "college", "degree", "gpa"),
            ("undergraduate", "graduate", "bachelor", "master", "course", "class"),
        ),
        (
            ("prefer", "recommend", "suggest", "choose", "like"),
            ("preference", "favorite", "dislike", "constraint", "goal", "habit"),
        ),
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for triggers, expansions in groups:
        if not any(trigger in lowered for trigger in triggers):
            continue
        for term in expansions:
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms


def strategy_lexical_top_k(strategy: QuestionStrategy, default: int) -> int:
    if strategy.name in {"multi_evidence", "temporal"}:
        return max(default, strategy.top_k)
    if strategy.name in {"preference", "recency"}:
        return max(default, 60)
    return default


def prefer_user_chunks(
    example: Example,
    strategy: QuestionStrategy,
    retrieved: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    if asks_assistant_memory(example.question) or strategy.name not in {"multi_evidence", "preference"}:
        return rerank_retrieved(retrieved[:top_k])
    user_chunks = [item for item in retrieved if item.text.lstrip().lower().startswith("user:")]
    other_chunks = [item for item in retrieved if item not in user_chunks]
    return rerank_retrieved([*user_chunks[:top_k], *other_chunks][:top_k])


def select_retrieved_by_ranks(candidates: list[RetrievedChunk], selected_ranks: list[int], keep: int) -> list[RetrievedChunk]:
    if keep <= 0 or not candidates:
        return candidates
    by_rank = {int(candidate.rank): candidate for candidate in candidates}
    selected = []
    seen = set()
    for rank in selected_ranks:
        candidate = by_rank.get(rank)
        if candidate is None or candidate.chunk_id in seen:
            continue
        selected.append(candidate)
        seen.add(candidate.chunk_id)
        if len(selected) >= keep:
            break
    if not selected:
        selected = candidates[:keep]
    return rerank_retrieved(selected)


def rerank_retrieved(items: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=item.chunk_id,
            text=item.text,
            date=item.date,
            session_id=item.session_id,
            score=item.score,
            rank=rank,
        )
        for rank, item in enumerate(items, start=1)
    ]

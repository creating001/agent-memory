from __future__ import annotations

import re
from typing import Any

from agent_memory.core.schema import Example, RetrievedChunk
from agent_memory.baseline.routing import QuestionStrategy, asks_assistant_memory, is_collection_intent_question


def embedding_text(chunk: Any, *, mode: str = "date_content") -> str:
    if mode == "content_only":
        return str(chunk.text)
    if mode == "metadata_content":
        parts = []
        if chunk.date:
            parts.append(f"Date: {chunk.date}")
        if getattr(chunk, "session_id", ""):
            parts.append(f"Session: {chunk.session_id}")
        parts.append(f"Content: {chunk.text}")
        return "\n".join(parts)
    if chunk.date:
        return f"Date: {chunk.date}\nContent: {chunk.text}"
    return chunk.text


def retrieval_query_text(
    example: Example,
    *,
    temporal_hints: list[str] | None = None,
    include_question_date: bool = True,
) -> str:
    suffix = ""
    if temporal_hints:
        suffix = "\nTemporal search hints: " + "; ".join(temporal_hints)
    if include_question_date and example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}{suffix}"
    return f"{example.question}{suffix}"


def expanded_retrieval_query_text(
    example: Example,
    *,
    temporal_hints: list[str] | None = None,
    include_question_date: bool = True,
) -> str:
    question = retrieval_query_text(
        example,
        temporal_hints=temporal_hints,
        include_question_date=include_question_date,
    )
    return (
        f"{question}\n\n"
        "Find memories that provide direct evidence for this question, including relevant facts, dates, "
        "numbers, events, names, user preferences, dislikes, constraints, owned resources, prior experiences, "
        "habits, and goals."
    )


def expanded_lexical_query_text(
    example: Example,
    *,
    temporal_hints: list[str] | None = None,
    include_question_date: bool = True,
    collection_query_expansion: bool = False,
    scope_aware_collection_query_expansion: bool = False,
) -> str:
    question = retrieval_query_text(
        example,
        temporal_hints=temporal_hints,
        include_question_date=include_question_date,
    )
    terms = lexical_expansion_terms(
        example.question,
        collection_query_expansion=collection_query_expansion,
        scope_aware_collection_query_expansion=scope_aware_collection_query_expansion,
    )
    if not terms:
        return question
    return f"{question}\nRelated terms: {' '.join(terms)}"


def lexical_expansion_terms(
    question: str,
    *,
    collection_query_expansion: bool = False,
    scope_aware_collection_query_expansion: bool = False,
) -> list[str]:
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
    if collection_query_expansion and is_collection_intent_question(lowered):
        for term in collection_query_expansion_terms(lowered, scope_aware=scope_aware_collection_query_expansion):
            if term not in seen:
                terms.append(term)
                seen.add(term)
    return terms


def collection_query_expansion_terms(lowered: str, *, scope_aware: bool = False) -> list[str]:
    if scope_aware:
        return scope_aware_collection_query_expansion_terms(lowered)

    groups = [
        (
            ("name", "names", "people", "person"),
            (
                "named",
                "called",
                "friend",
                "family",
                "child",
                "children",
                "kid",
                "kids",
                "son",
                "daughter",
                "partner",
                "coworker",
                "colleague",
            ),
        ),
        (
            ("pet", "pets", "dog", "cat", "animal"),
            (
                "pet",
                "pets",
                "dog",
                "cat",
                "puppy",
                "pup",
                "kitty",
                "turtle",
                "named",
                "adopted",
                "shelter",
            ),
        ),
        (
            ("city", "cities", "state", "states", "country", "countries", "location", "locations"),
            (
                "travel",
                "trip",
                "visited",
                "went",
                "vacation",
                "city",
                "cities",
                "state",
                "country",
                "location",
                "moved",
            ),
        ),
        (
            ("activities", "activity", "interests", "subjects", "hobbies", "hobby"),
            (
                "activity",
                "activities",
                "hobby",
                "hobbies",
                "class",
                "workshop",
                "practice",
                "volunteer",
                "running",
                "biking",
                "gardening",
                "surfing",
                "yoga",
                "pottery",
                "painting",
                "art",
                "show",
                "concert",
            ),
        ),
        (
            ("books", "book", "authors", "author"),
            ("book", "books", "read", "author", "authors", "series", "novel", "recommended"),
        ),
        (
            ("items", "item", "damages", "damage", "shelters", "shelter"),
            ("item", "items", "owned", "bought", "purchased", "broke", "damaged", "repair", "shelter"),
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


def scope_aware_collection_query_expansion_terms(lowered: str) -> list[str]:
    terms: list[str] = []

    def add(values: tuple[str, ...]) -> None:
        for term in values:
            if term not in terms:
                terms.append(term)

    travel_terms = ("travel", "trip", "visited", "went", "vacation", "moved")
    if any(signal in lowered for signal in ("u.s. cities", "us cities", "which cities", "which city", "what cities")):
        add((*travel_terms, "city", "cities"))
    elif "states" in lowered or "state" in lowered:
        add((*travel_terms, "state", "states"))
    elif "countries" in lowered or "country" in lowered:
        add((*travel_terms, "country", "countries"))
    elif "geographical locations" in lowered or "locations" in lowered or "location" in lowered:
        add((*travel_terms, "city", "cities", "state", "states", "country", "countries", "location", "locations"))

    groups = [
        (
            ("name", "names", "people", "person"),
            (
                "named",
                "called",
                "friend",
                "family",
                "child",
                "children",
                "kid",
                "kids",
                "son",
                "daughter",
                "partner",
                "coworker",
                "colleague",
            ),
        ),
        (
            ("pet", "pets", "dog", "cat", "animal"),
            (
                "pet",
                "pets",
                "dog",
                "cat",
                "puppy",
                "pup",
                "kitty",
                "turtle",
                "named",
                "adopted",
                "shelter",
            ),
        ),
        (
            ("activities", "activity", "interests", "subjects", "hobbies", "hobby"),
            (
                "activity",
                "activities",
                "hobby",
                "hobbies",
                "class",
                "workshop",
                "practice",
                "volunteer",
                "running",
                "biking",
                "gardening",
                "surfing",
                "pottery",
                "painting",
                "art",
                "show",
                "concert",
            ),
        ),
        (
            ("books", "book", "authors", "author"),
            ("book", "books", "read", "author", "authors", "series", "novel", "recommended"),
        ),
        (
            ("items", "item", "damages", "damage", "shelters", "shelter"),
            ("item", "items", "owned", "bought", "purchased", "broke", "damaged", "repair", "shelter"),
        ),
    ]
    excluded = excluded_scope_terms(lowered)
    for triggers, expansions in groups:
        if not any(trigger in lowered for trigger in triggers):
            continue
        for term in expansions:
            if term in terms or is_excluded_expansion_term(term, excluded):
                continue
            terms.append(term)
    return terms


def is_excluded_expansion_term(term: str, excluded: set[str]) -> bool:
    normalized = "".join(ch for ch in term.lower() if ch.isalnum())
    if not normalized:
        return False
    stems = {normalized}
    if normalized.endswith("e") and len(normalized) > 4:
        stems.add(normalized[:-1])
    for value in excluded:
        value_stems = {value}
        if value.endswith("ing") and len(value) > 5:
            value_stems.add(value[:-3])
        if any(left.startswith(right) or right.startswith(left) for left in stems for right in value_stems):
            return True
    return False


def excluded_scope_terms(lowered: str) -> set[str]:
    phrases = []
    for signal in ("besides", "other than", "except", "excluding"):
        marker = f" {signal} "
        if marker not in f" {lowered} ":
            continue
        phrase = lowered.split(signal, 1)[1]
        phrase = phrase.split("?", 1)[0].split(",", 1)[0].split(";", 1)[0]
        phrases.append(phrase)
    excluded: set[str] = set()
    for phrase in phrases:
        for term in phrase.replace("and", " ").split():
            normalized = "".join(ch for ch in term if ch.isalnum())
            if len(normalized) >= 3:
                excluded.add(normalized)
    return excluded


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
    *,
    non_user_keep: int = 0,
) -> list[RetrievedChunk]:
    if asks_assistant_memory(example.question) or strategy.name not in {"multi_evidence", "preference"}:
        return rerank_retrieved(retrieved[:top_k])
    if non_user_keep <= 0:
        user_chunks = [item for item in retrieved if is_user_memory_chunk(item)]
        other_chunks = [item for item in retrieved if item not in user_chunks]
        return rerank_retrieved([*user_chunks[:top_k], *other_chunks][:top_k])

    user_limit = max(0, top_k - min(non_user_keep, top_k))
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()

    def add(item: RetrievedChunk) -> None:
        if item.chunk_id in seen or len(selected) >= top_k:
            return
        selected.append(item)
        seen.add(item.chunk_id)

    user_chunks = [item for item in retrieved if is_user_memory_chunk(item)]
    other_chunks = [item for item in retrieved if item not in user_chunks]
    for item in user_chunks[:user_limit]:
        add(item)
    for item in other_chunks[:non_user_keep]:
        add(item)
    for item in retrieved:
        add(item)
    return rerank_retrieved(selected)


def is_user_memory_chunk(item: RetrievedChunk) -> bool:
    text = item.text.lstrip().lower()
    return text.startswith("user:")


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

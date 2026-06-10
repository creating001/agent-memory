from __future__ import annotations

import re
from typing import Any

from agent_memory.baseline.config_aliases import normalize_rerank_gate
from agent_memory.baseline.queries import rerank_retrieved
from agent_memory.baseline.routing import QuestionStrategy
from agent_memory.core.schema import Example, RetrievedChunk


def apply_dedicated_rerank_anchor_retention(
    *,
    original: list[RetrievedChunk],
    reranked: list[RetrievedChunk],
    rerank_cfg: dict[str, Any],
) -> list[RetrievedChunk]:
    anchor_keep = max(0, int(rerank_cfg.get("anchor_keep", 0)))
    if anchor_keep <= 0 or not original or not reranked:
        return reranked

    anchor_after_top = max(0, int(rerank_cfg.get("anchor_after_top", 0)))
    selected: list[RetrievedChunk] = []
    seen: set[str] = set()

    def add(item: RetrievedChunk) -> None:
        if item.chunk_id in seen:
            return
        selected.append(item)
        seen.add(item.chunk_id)

    for item in reranked[:anchor_after_top]:
        add(item)
    for item in original[:anchor_keep]:
        add(item)
    for item in reranked:
        add(item)
    return rerank_retrieved(selected)


def should_use_dedicated_rerank(
    config: dict[str, Any],
    strategy: QuestionStrategy,
    example: Example | None = None,
    route_trace: dict[str, Any] | None = None,
) -> bool:
    rerank_cfg = config.get("rerank") or {}
    if not bool(rerank_cfg.get("enabled", False)):
        return False
    strategies = rerank_cfg.get("strategies")
    if strategies and strategy.name not in {str(value) for value in strategies}:
        return False

    gate = normalize_rerank_gate(str(rerank_cfg.get("gate", "all")))
    if gate == "all":
        return True
    if gate == "simple_factual_slots_v1":
        return example is not None and is_simple_factual_slot_question(example.question)
    if gate == "simple_factual_slots_v2":
        return example is not None and is_simple_factual_slot_question(
            example.question,
            include_action_slots=True,
        )
    if gate == "simple_factual_slots_v2_or_multi_evidence":
        if strategy.name == "multi_evidence":
            return True
        return example is not None and is_simple_factual_slot_question(
            example.question,
            include_action_slots=True,
        )
    if gate == "simple_factual_slots_v3":
        return example is not None and is_simple_factual_slot_question(
            example.question,
            include_action_slots=True,
            action_slot_mode="past_or_singular",
        )
    if gate == "simple_factual_slots_v3_or_temporal":
        if strategy.name == "temporal":
            return True
        return example is not None and is_simple_factual_slot_question(
            example.question,
            include_action_slots=True,
            action_slot_mode="past_or_singular",
        )
    if gate == "semantic_single_fact":
        return is_semantic_single_fact_route(route_trace)
    return False


def dedicated_rerank_pool_k(retrieval_cfg: dict[str, Any], strategy: QuestionStrategy) -> int:
    pool_k = int(retrieval_cfg.get("dedicated_rerank_pool_k", 0) or 0)
    overrides = retrieval_cfg.get("dedicated_rerank_pool_k_by_strategy") or {}
    if not isinstance(overrides, dict):
        return pool_k
    raw_value = overrides.get(strategy.name)
    if raw_value is None:
        return pool_k
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return pool_k


def is_semantic_single_fact_route(route_trace: dict[str, Any] | None) -> bool:
    """Use the LLM router's general task schema instead of hand-written question signals."""
    if not route_trace:
        return False
    task = str(route_trace.get("task") or "")
    operation = str(route_trace.get("operation") or "")
    answer_slot = str(route_trace.get("answer_slot") or "")
    temporal_subtype = str(route_trace.get("temporal_subtype") or "")
    if task != "single_fact":
        return False
    if operation not in {"none", ""}:
        return False
    if temporal_subtype not in {"none", ""}:
        return False
    return answer_slot not in {"count", "list", "duration", "preference"}


def is_simple_factual_slot_question(
    question: str,
    *,
    include_action_slots: bool = False,
    action_slot_mode: str = "all",
) -> bool:
    """Gate rerank to narrow slot questions where cross-encoder precision is helpful."""
    lowered = re.sub(r"\s+", " ", question.lower()).strip()
    broad_or_inference_signals = (
        "what are ",
        "what were ",
        "what kind of ",
        "what kinds of ",
        "what type of ",
        "what types of ",
        "what style of ",
        "what places ",
        "what things ",
        "what items ",
        "what activities ",
        "what events ",
        "what books ",
        "what games ",
        "what bands ",
        "what artists",
        "what symbols ",
        "who have ",
        "who or which ",
        "which places ",
        "which bands ",
        "which of ",
        "shared ",
        "similar ",
        "likely ",
        "might ",
        "could ",
        "would ",
        "besides ",
        "other than ",
        "how often ",
        "how did ",
        "how many ",
        "how long ",
        "what can ",
    )
    if any(signal in lowered for signal in broad_or_inference_signals):
        return False

    if any(marker in lowered for marker in (" from whom", " to whom", " with whom")):
        return True
    if lowered.startswith(("what is the ", "what was the ", "what's the ", "what new ")):
        return True
    if lowered.startswith(("which year ", "which month ", "which day ", "which date ", "which book ", "which novel ")):
        return True
    if lowered.startswith(("where do ", "where does ", "where is ", "where are ", "where was ", "where were ")):
        return True
    if not include_action_slots:
        return False
    if action_slot_mode == "past_or_singular":
        return lowered.startswith(("what did ", "what does "))
    return lowered.startswith(("what did ", "what do ", "what does "))


def format_rerank_document(item: RetrievedChunk, *, max_chars: int = 0) -> str:
    if item.date:
        document = f"Date: {item.date}\nContent: {item.text}"
    else:
        document = item.text
    return truncate_for_rerank(document, max_chars=max_chars)


def truncate_for_rerank(document: str, *, max_chars: int = 0) -> str:
    if max_chars <= 0 or len(document) <= max_chars:
        return document
    marker = "\n...\n"
    if max_chars <= len(marker) + 64:
        return document[:max_chars]
    side = (max_chars - len(marker)) // 2
    return f"{document[:side]}{marker}{document[-(max_chars - len(marker) - side):]}"

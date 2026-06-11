from __future__ import annotations

import re
from typing import Any

from agent_memory.baseline import intent_features as features
from agent_memory.baseline.config_aliases import (
    normalize_duration_evidence_mode,
    normalize_multi_evidence_prompt_mode,
    normalize_router_mode,
)
from agent_memory.baseline.guardrails import (
    deterministic_relative_event_order_answer,
    is_insufficient_answer,
    router_required_target_missing_reason,
    user_memory_context,
)
from agent_memory.baseline.routing import (
    QuestionStrategy,
    RouteSettings,
    asks_explicit_assistant_memory,
    collection_intent_mode,
    is_collection_intent_question,
    is_generic_temporal_question,
    is_multi_evidence_question,
    is_strict_temporal_question,
    use_duration_evidence_requirements_question,
)
from agent_memory.core.schema import Example


def answer_guardrail_options(answer_cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize public guardrail config names to the internal guardrail API."""

    def cfg(clean_key: str, legacy_key: str, default: Any) -> Any:
        return answer_cfg.get(clean_key, answer_cfg.get(legacy_key, default))

    return {
        "relative_temporal_guardrail": bool(answer_cfg.get("relative_temporal_guardrail", False)),
        "relative_temporal_guardrail_scope": str(answer_cfg.get("relative_temporal_guardrail_scope", "context")),
        "relative_temporal_guardrail_loose": bool(answer_cfg.get("relative_temporal_guardrail_loose", False)),
        "relative_temporal_guardrail_require_temporal_hypothesis": bool(
            answer_cfg.get("relative_temporal_guardrail_require_temporal_hypothesis", False)
        ),
        "relative_temporal_guardrail_allow_insufficient_year": bool(
            answer_cfg.get("relative_temporal_guardrail_allow_insufficient_year", False)
        ),
        "relative_temporal_guardrail_preserve_relative_weekday": bool(
            answer_cfg.get("relative_temporal_guardrail_preserve_relative_weekday", False)
        ),
        "relative_temporal_guardrail_future_relative": bool(
            answer_cfg.get("relative_temporal_guardrail_future_relative", False)
        ),
        "relative_temporal_guardrail_allow_excluded_relative": bool(
            answer_cfg.get("relative_temporal_guardrail_allow_excluded_relative", False)
        ),
        "relative_temporal_guardrail_allow_insufficient_included_relative": bool(
            answer_cfg.get("relative_temporal_guardrail_allow_insufficient_included_relative", False)
        ),
        "missing_target_guardrail": bool(cfg("target_consistency_guardrail", "missing_target_guardrail", True)),
        "missing_target_guardrail_skip_inference": bool(
            cfg(
                "target_consistency_skip_inference_questions",
                "target_consistency_skip_unsupported_inference",
                answer_cfg.get("missing_target_guardrail_skip_inference", False),
            )
        ),
        "missing_target_guardrail_skip_family_answered": bool(
            cfg("target_consistency_skip_family_answered", "missing_target_guardrail_skip_family_answered", False)
        ),
        "missing_target_guardrail_relation_aliases": bool(
            cfg("target_consistency_relation_normalization", "missing_target_guardrail_relation_aliases", False)
        ),
    }


def final_max_tokens(answer_cfg: dict[str, Any]) -> int:
    configured = int(answer_cfg.get("max_tokens", 8192))
    return int(answer_cfg.get("final_max_tokens", min(configured, 1024)))


def answer_max_context_chars(answer_cfg: dict[str, Any], strategy: QuestionStrategy) -> int:
    default = int(answer_cfg.get("max_context_chars", 12000))
    overrides = answer_cfg.get("max_context_chars_by_strategy") or {}
    if not isinstance(overrides, dict):
        return default
    raw_value = overrides.get(strategy.name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def is_ordinal_lookup_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(\d+(?:st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)\b",
            question,
            flags=re.IGNORECASE,
        )
    )


def map_route_plan(plan: dict[str, Any], retrieval_cfg: dict[str, Any] | None = None) -> tuple[str, str]:
    task = str(plan.get("task") or "single_fact")
    operation = str(plan.get("operation") or "none")
    temporal_subtype = str(plan.get("temporal_subtype") or "none")
    retrieval_cfg = retrieval_cfg or {}
    aggregation_operations = {"count", "sum", "compare", "list", "set_relation"}
    aggregation_requires_multi_task = bool(
        retrieval_cfg.get("semantic_router_aggregation_requires_multi_task", False)
    )
    list_requires_multi_task = bool(
        retrieval_cfg.get("semantic_router_list_requires_multi_task", False)
    )
    operation_requires_multi_task = aggregation_requires_multi_task or (
        list_requires_multi_task and operation in {"list", "set_relation"}
    )

    if task == "preference_advice" or operation == "preference":
        return "prefer_user", "preference"
    if task == "multi_evidence" or (
        operation in aggregation_operations and not operation_requires_multi_task
    ):
        route = str(retrieval_cfg.get("semantic_router_multi_evidence_route", "auto"))
        operation_routes = retrieval_cfg.get("semantic_router_multi_evidence_route_by_operation") or {}
        if isinstance(operation_routes, dict):
            route = str(operation_routes.get(operation) or route)
        excluded_answer_slots = {
            str(value) for value in retrieval_cfg.get("semantic_router_multi_evidence_route_excluded_answer_slots", [])
        }
        if excluded_answer_slots and str(plan.get("answer_slot") or "") in excluded_answer_slots:
            route = str(retrieval_cfg.get("semantic_router_multi_evidence_route", "auto"))
        if bool(retrieval_cfg.get("semantic_router_multi_evidence_route_exclude_ordinal_lookup", False)):
            question = str(plan.get("_question") or "")
            if operation == "list" and is_ordinal_lookup_question(question):
                route = str(retrieval_cfg.get("semantic_router_multi_evidence_route", "auto"))
        if route not in {"auto", "prefer_user"}:
            route = "auto"
        return route, "multi_evidence"
    if task == "temporal" or operation in {"duration", "order", "date_time"}:
        if temporal_subtype == "duration" or operation == "duration":
            return "duration_temporal", "temporal"
        if temporal_subtype == "order" or operation == "order":
            return "order_reflective", "temporal"
        return "auto", "temporal"
    if task == "recency_state" or operation == "recency":
        return "recency_auto", "recency"
    return "auto", "factual"


def use_temporal_prompt(config: dict[str, Any], strategy: QuestionStrategy, settings: RouteSettings) -> bool:
    return bool(
        config["answer"].get("temporal_prompt", False)
        and strategy.name == "temporal"
        and settings.use_evidence_table
        and strategy.use_evidence_table
    )


def uses_llm_task_router(config: dict[str, Any]) -> bool:
    mode = normalize_router_mode(str((config.get("retrieval") or {}).get("router_mode", "legacy")))
    return mode in {"semantic_task", "selective_semantic_task"}


def should_use_semantic_router_prompt_trace(config: dict[str, Any]) -> bool:
    """Let the router's task schema guide prompts without changing route or strategy."""
    retrieval_cfg = config.get("retrieval") or {}
    return uses_llm_task_router(config) and bool(retrieval_cfg.get("semantic_router_prompt_trace", False))


def should_use_duration_evidence_prompt(
    config: dict[str, Any],
    question: str,
    route_plan: dict[str, Any] | None,
) -> bool:
    if uses_llm_task_router(config) and route_plan:
        return bool(
            (
                route_plan.get("temporal_subtype") == "duration"
                or route_plan.get("operation") == "duration"
                or route_plan.get("answer_slot") == "duration"
            )
        )
    return use_duration_evidence_requirements_question(
        question,
        mode=normalize_duration_evidence_mode(str(config["answer"].get("duration_evidence_requirements_mode", "all"))),
    )


def should_use_list_evidence_prompt(
    config: dict[str, Any],
    question: str,
    route_plan: dict[str, Any] | None,
) -> bool:
    if uses_llm_task_router(config) and route_plan:
        return bool(
            (
                route_plan.get("answer_slot") == "list"
                or route_plan.get("operation") in {"list", "set_relation"}
            )
        )
    return is_collection_intent_question(
        question.lower(),
        mode=collection_intent_mode(config),
    )


def should_force_strict_multi_evidence(
    answer_cfg: dict[str, Any],
    question: str,
    strategy: QuestionStrategy,
) -> bool:
    if strategy.name != "multi_evidence" or not bool(answer_cfg.get("strict_multi_evidence", False)):
        return False
    mode = str(answer_cfg.get("strict_multi_evidence_mode", "all"))
    if mode == "all":
        return True
    if mode == "count_like":
        lowered = question.lower()
        return is_multi_evidence_question(lowered) and not is_strict_temporal_question(lowered)
    return False


def should_force_evidence_table(answer_cfg: dict[str, Any], strategy: QuestionStrategy) -> bool:
    strategies = answer_cfg.get("force_evidence_table_strategies", [])
    if not strategies:
        return False
    return strategy.name in {str(value) for value in strategies}


def should_use_answer_detail_requirements(
    answer_cfg: dict[str, Any],
    example: Example,
    *,
    route_plan: dict[str, Any] | None,
    route_trace: dict[str, Any] | None,
) -> bool:
    if not bool(answer_cfg.get("answer_detail_requirements", False)):
        return False
    mode = str(answer_cfg.get("answer_detail_requirements_mode", "all"))
    if mode == "all":
        return True
    if mode != "operation_v1":
        return False

    plan = route_trace or route_plan or {}
    task = str(plan.get("task") or "")
    operation = str(plan.get("operation") or "")
    answer_slot = str(plan.get("answer_slot") or "")

    if asks_explicit_assistant_memory(example.question):
        return True
    if operation in {"sum", "duration", "date_time"}:
        return True
    if task == "single_fact" and operation == "list" and answer_slot == "list":
        return True
    return False


def should_include_router_target_support(answer_cfg: dict[str, Any], retrieval_cfg: dict[str, Any]) -> bool:
    if bool(retrieval_cfg.get("semantic_router_include_target_support", False)):
        return True
    return bool(answer_cfg.get("target_consistency_verify", False) or answer_cfg.get("router_target_phrase_guard", False))


def router_target_phrase_missing_candidate(
    config: dict[str, Any],
    example: Example,
    chunks: list[Any],
    route_trace: dict[str, Any] | None,
    strategy: QuestionStrategy,
    hypothesis: str,
) -> str | None:
    answer_cfg = config.get("answer") or {}
    if not bool(answer_cfg.get("router_target_phrase_guard", False)):
        return None
    if is_insufficient_answer(hypothesis):
        return None
    if not route_trace or not route_trace.get("target_support_required"):
        return None
    guard_strategies = answer_cfg.get("router_target_phrase_guard_strategies", [])
    if guard_strategies and strategy.name not in {str(value) for value in guard_strategies}:
        return None
    guard_operations = answer_cfg.get("router_target_phrase_guard_operations", [])
    operation = str(route_trace.get("operation") or "")
    if guard_operations and operation not in {str(value) for value in guard_operations}:
        return None
    if operation == "order":
        if " or " not in f" {example.question.lower()} ":
            return None
        if deterministic_relative_event_order_answer(example.question, chunks):
            return None
    target_phrases = route_trace.get("required_target_phrases") or []
    if not target_phrases:
        return None
    context = "\n".join(str(getattr(chunk, "text", "") or "") for chunk in chunks)
    if not asks_explicit_assistant_memory(example.question):
        user_context = user_memory_context(chunks)
        if user_context:
            context = user_context
    return router_required_target_missing_reason(target_phrases, context)


def should_force_direct_answer(answer_cfg: dict[str, Any], strategy: QuestionStrategy) -> bool:
    strategies = answer_cfg.get("direct_answer_strategies", [])
    if not strategies:
        return False
    return strategy.name in {str(value) for value in strategies}


def should_use_context_relative_time_annotations(
    answer_cfg: dict[str, Any],
    strategy: QuestionStrategy,
    question: str,
) -> bool:
    if not bool(answer_cfg.get("context_relative_time_annotations", False)):
        return False
    strategies = answer_cfg.get("context_relative_time_annotation_strategies", [])
    if strategies and strategy.name not in {str(value) for value in strategies}:
        return False
    question_mode = str(answer_cfg.get("context_relative_time_annotation_question_mode", "all"))
    if question_mode == "generic_temporal":
        return is_generic_temporal_question(question.lower())
    return True


def temporal_query_hints(config: dict[str, Any], route_trace: dict[str, Any] | None) -> list[str]:
    retrieval_cfg = config.get("retrieval") or {}
    if not bool(retrieval_cfg.get("temporal_query_anchors", False)):
        return []
    if not route_trace:
        return []
    hints = route_trace.get("temporal_search_hints") or []
    if not isinstance(hints, list):
        return []
    require_normalized = bool(retrieval_cfg.get("temporal_query_anchor_require_normalized", False))
    cleaned = []
    for hint in hints:
        text = " ".join(str(hint).split())
        if require_normalized and not looks_like_normalized_temporal_hint(text):
            continue
        if text:
            cleaned.append(text[:120])
        if len(cleaned) >= 6:
            break
    return cleaned


def should_include_question_date_in_query(
    config: dict[str, Any],
    strategy: QuestionStrategy,
    route: str,
) -> bool:
    retrieval_cfg = config.get("retrieval") or {}
    mode = str(retrieval_cfg.get("question_date_query_mode", "always"))
    if mode == "never":
        return False
    if mode != "strategy_allowlist":
        return True

    allowed_strategies = {str(value) for value in retrieval_cfg.get("question_date_query_strategies", [])}
    if strategy.name in allowed_strategies:
        return True

    default_temporal_routes = {
        "duration_temporal",
        "order_reflective",
        "order_date_rewrite",
        "state_history",
        "recency_auto",
    }
    configured_routes = retrieval_cfg.get("question_date_query_routes")
    allowed_routes = (
        {str(value) for value in configured_routes}
        if configured_routes is not None
        else default_temporal_routes
    )
    return route in allowed_routes


def looks_like_normalized_temporal_hint(text: str) -> bool:
    if not text:
        return False
    if re.search(r"\b\d{4}(?:[-/]\d{1,2}){0,2}\b", text):
        return True
    month = (
        r"january|february|march|april|may|june|july|august|"
        r"september|october|november|december"
    )
    return bool(re.search(rf"\b(?:{month})\b.*\b\d{{4}}\b|\b\d{{4}}\b.*\b(?:{month})\b", text, re.IGNORECASE))


def should_use_factual_lexical(config: dict[str, Any], strategy: QuestionStrategy, route: str) -> bool:
    retrieval_cfg = config.get("retrieval") or {}
    if strategy.name != "factual" or not bool(retrieval_cfg.get("factual_lexical_retrieve", False)):
        return False
    routes = {str(value) for value in retrieval_cfg.get("factual_lexical_routes", [])}
    return not routes or route in routes


def effective_multi_evidence_prompt_mode(
    answer_cfg: dict[str, Any],
    question: str,
    *,
    route_plan: dict[str, Any] | None = None,
) -> str:
    mode = normalize_multi_evidence_prompt_mode(str(answer_cfg.get("multi_evidence_prompt", "default")))
    if mode == "set_operation_selective":
        return "set_operation" if is_set_operation_question_without_advice_frame(question) else "scoped_aggregation"
    if mode in {"set_operation", "scoped_aggregation"}:
        return mode
    return "scoped_aggregation"


def is_set_operation_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if re.search(r"\b(did|do|does|have|has|what|which|where|who|are|were)\b.*\bboth\b", normalized):
        return True
    if "have in common" in normalized:
        return True
    if re.search(r"\b(shared|common)\b", normalized):
        return True
    return bool(re.search(r"\bsimilar\b", normalized) and re.search(r"\b(pair|item|object|thing|place|activity)\b", normalized))


def is_set_operation_question_without_advice_frame(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if any(signal in normalized for signal in features.SET_OPERATION_ADVICE_EXCLUSION_CUES):
        return False
    return is_set_operation_question(question)

"""Question intent routing for the memory pipeline.

The router maps question semantics to retrieval and answer-generation
strategies. It is intentionally dataset-agnostic: routing only depends on the
question text and runtime configuration, never on dataset names, sample ids,
gold answers, or judge feedback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from agent_memory.baseline import intent_features as features
from agent_memory.baseline.config_aliases import (
    normalize_collection_intent_mode,
    normalize_preference_inference_mode,
    normalize_router_mode,
    normalize_strategy_profile,
)
from agent_memory.core.schema import Example


METHOD_NAME = "strong_memory_baseline_v1"


@dataclass(frozen=True)
class QuestionStrategy:
    name: str
    use_expanded_query: bool
    use_lexical: bool
    use_reflective: bool
    use_rerank: bool
    use_evidence_table: bool
    requirement_style: str
    top_k: int


@dataclass(frozen=True)
class RouteSettings:
    strategy_profile: str
    top_k: int
    strategy_top_k: dict[str, int] = field(default_factory=dict)
    prefer_user_chunks: bool = False
    use_evidence_rerank: bool = True
    use_reflective_retrieval: bool = False
    use_evidence_table: bool = True
    use_verification: bool = False
    verification_strategies: frozenset[str] = frozenset()
    context_window: int = 0
    context_window_top_n: int = 0
    context_window_strategies: frozenset[str] = frozenset()
    use_temporal_order_compiler: bool = False


def choose_memory_route(example: Example, config: dict[str, Any] | None = None) -> str:
    if router_mode(config) == "surface_intent":
        return choose_memory_route_generic(example, config)

    question = example.question
    route = "auto"
    enumerable_intent = use_collection_intent_routing(config) and is_collection_intent_question(
        question.lower(),
        mode=collection_intent_mode(config),
    )
    base_strategy = choose_strategy(
        question,
        profile=configured_profile("general", config),
        collection_intent_routing=use_collection_intent_routing(config),
        personalized_inference_routing=use_personalized_inference_routing(config),
        preference_inference_mode=preference_inference_mode(config),
        routing_mode=router_mode(config),
        collection_intent_mode=collection_intent_mode(config),
    )
    if (
        base_strategy.name in {"multi_evidence", "preference"}
        and not asks_assistant_memory(question)
        and not is_temporal_comparison_quantity_question(question)
        and not enumerable_intent
        and not should_keep_auto_route(question, config)
    ):
        route = "prefer_user"
    if is_order_route_question(question):
        route = "order_reflective"
    if is_duration_route_question(question):
        route = "duration_temporal"
    if is_historical_state_question(question):
        route = "state_history"
    if is_most_recent_question(question):
        route = "recency_auto"
    return route


def choose_memory_route_generic(example: Example, config: dict[str, Any] | None = None) -> str:
    question = example.question
    lowered = question.lower()
    route = "auto"
    enumerable_intent = use_collection_intent_routing(config) and is_collection_intent_question(
        lowered,
        mode=collection_intent_mode(config),
    )
    base_strategy = choose_strategy(
        question,
        profile=configured_profile("general", config),
        collection_intent_routing=use_collection_intent_routing(config),
        personalized_inference_routing=use_personalized_inference_routing(config),
        preference_inference_mode=preference_inference_mode(config),
        routing_mode="surface_intent",
        collection_intent_mode=collection_intent_mode(config),
    )
    if (
        base_strategy.name in {"multi_evidence", "preference"}
        and not asks_assistant_memory(question)
        and not is_generic_temporal_comparison_quantity_question(lowered)
        and not enumerable_intent
        and not should_keep_auto_route(question, config)
    ):
        route = "prefer_user"
    if is_generic_order_route_question(lowered):
        route = "order_reflective"
    if is_generic_duration_route_question(lowered):
        route = "duration_temporal"
    if is_generic_historical_state_question(lowered) and not asks_assistant_memory(question):
        route = "state_history"
    if is_generic_most_recent_question(lowered):
        route = "recency_auto"
    return route


def route_settings(route: str, config: dict[str, Any]) -> RouteSettings:
    default_top_k = int(config["retrieval"].get("top_k", 40))
    context_window = int(config["retrieval"].get("context_window", 0))
    context_window_top_n = int(config["retrieval"].get("context_window_top_n", 0))
    context_window_strategies = frozenset(
        str(value) for value in config["retrieval"].get("context_window_strategies", [])
    )
    def profile(name: str) -> str:
        return configured_profile(name, config)

    def configured_verification() -> bool:
        routes = {str(value) for value in config.get("answer", {}).get("verification_routes", [])}
        return "all" in routes or route in routes

    def configured_verification_strategies() -> frozenset[str]:
        values = config.get("answer", {}).get("verification_strategies", [])
        if not values:
            return frozenset({"factual", "multi_evidence", "preference", "recency", "temporal"})
        return frozenset(str(value) for value in values)

    if route == "prefer_user":
        prefer_user_top_k = int(config["retrieval"].get("prefer_user_top_k", 40))
        return RouteSettings(
            strategy_profile=profile("general"),
            top_k=prefer_user_top_k,
            prefer_user_chunks=True,
            use_evidence_rerank=False,
            use_verification=configured_verification(),
            verification_strategies=configured_verification_strategies(),
            context_window=context_window,
            context_window_top_n=context_window_top_n,
            context_window_strategies=context_window_strategies,
        )
    if route == "duration_temporal":
        return RouteSettings(
            strategy_profile=profile("balanced"),
            top_k=0,
            strategy_top_k={
                "factual": 40,
                "multi_evidence": 20,
                "preference": 40,
                "recency": 40,
                "temporal": 40,
            },
            use_verification=configured_verification(),
            verification_strategies=configured_verification_strategies(),
            context_window=context_window,
            context_window_top_n=context_window_top_n,
            context_window_strategies=context_window_strategies,
        )
    if route == "order_reflective":
        return RouteSettings(
            strategy_profile=profile("temporal_first"),
            top_k=40,
            use_reflective_retrieval=True,
            use_evidence_table=bool(config["retrieval"].get("order_reflective_use_evidence_table", False)),
            use_verification=True,
            verification_strategies=frozenset({"factual", "temporal"}),
            context_window=context_window,
            context_window_top_n=context_window_top_n,
            context_window_strategies=context_window_strategies,
            use_temporal_order_compiler=bool(config["answer"].get("temporal_order_compiler", False)),
        )
    if route == "state_history":
        return RouteSettings(
            strategy_profile=profile("temporal_first"),
            top_k=0,
            use_verification=configured_verification(),
            verification_strategies=configured_verification_strategies(),
            context_window=context_window,
            context_window_top_n=context_window_top_n,
            context_window_strategies=context_window_strategies,
        )
    if route == "recency_auto":
        return RouteSettings(
            strategy_profile=profile("temporal_first"),
            top_k=40,
            use_verification=configured_verification(),
            verification_strategies=configured_verification_strategies(),
            context_window=context_window,
            context_window_top_n=context_window_top_n,
            context_window_strategies=context_window_strategies,
        )
    return RouteSettings(
        strategy_profile=profile("general"),
        top_k=default_top_k,
        use_verification=configured_verification(),
        verification_strategies=configured_verification_strategies(),
        context_window=context_window,
        context_window_top_n=context_window_top_n,
        context_window_strategies=context_window_strategies,
    )


def choose_strategy(
    question: str,
    *,
    profile: str,
    collection_intent_routing: bool = False,
    personalized_inference_routing: bool = False,
    preference_inference_mode: str = "broad",
    routing_mode: str = "rule_based",
    collection_intent_mode: str = "specific",
) -> QuestionStrategy:
    lowered = question.lower()
    use_word_boundary = profile.endswith("_boundary")
    base_profile = normalize_strategy_profile(profile.removesuffix("_boundary"))
    if routing_mode == "surface_intent":
        if is_generic_preference_question(
            lowered,
            inference_signals=personalized_inference_routing,
            inference_mode=preference_inference_mode,
        ):
            return preference_strategy()
        if is_multi_evidence_question(
            lowered,
            word_boundary=use_word_boundary,
            collection_intent_routing=collection_intent_routing,
            collection_intent_mode=collection_intent_mode,
        ):
            return multi_evidence_strategy()
        if is_generic_temporal_question(lowered):
            return temporal_strategy()
        if is_generic_recency_question(lowered):
            return recency_strategy()
        return factual_strategy()
    if is_preference_question(
        lowered,
        inference_signals=personalized_inference_routing,
        inference_mode=preference_inference_mode,
    ):
        return preference_strategy()
    if base_profile == "general":
        if is_multi_evidence_question(
            lowered,
            word_boundary=use_word_boundary,
            collection_intent_routing=collection_intent_routing,
            collection_intent_mode=collection_intent_mode,
        ):
            return multi_evidence_strategy()
        if is_strict_temporal_question(lowered):
            return temporal_strategy()
        return factual_strategy()
    if base_profile == "balanced":
        if is_multi_evidence_question(
            lowered,
            word_boundary=use_word_boundary,
            collection_intent_routing=collection_intent_routing,
            collection_intent_mode=collection_intent_mode,
        ):
            return multi_evidence_strategy()
        if is_recency_question(lowered):
            return recency_strategy()
        if is_strict_temporal_question(lowered):
            return temporal_strategy()
        return factual_strategy()
    if is_temporal_question(lowered):
        return temporal_strategy()
    if is_multi_evidence_question(
        lowered,
        word_boundary=use_word_boundary,
        collection_intent_routing=collection_intent_routing,
        collection_intent_mode=collection_intent_mode,
    ):
        return multi_evidence_strategy()
    if is_recency_question(lowered):
        return recency_strategy()
    return factual_strategy()


def effective_top_k(configured_top_k: int, strategy: QuestionStrategy, overrides: dict[str, int]) -> int:
    if strategy.name in overrides:
        return overrides[strategy.name]
    if configured_top_k <= 0:
        return strategy.top_k
    return min(configured_top_k, strategy.top_k)


def should_use_strict_multi_aggregation(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in features.STRICT_MULTI_AGGREGATION_PATTERNS)


def asks_explicit_assistant_memory(question: str) -> bool:
    return asks_assistant_memory(question)


def is_recommendation_recall_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if normalized.startswith(("can you ", "could you ", "would you ", "do you ", "should i ")):
        return False
    if normalized.startswith(("what should i ", "which should i ")):
        return False
    if normalized.startswith("what advice "):
        return True
    if re.search(r"^(?:what|which)\b.*\b(?:recommend|recommended|suggest|suggested)\b", normalized):
        return True
    if re.search(
        r"^(?:what|which)\b.*\b(?:recommendations?|suggestions?)\b.*\b(?:given|received|from|to|for|has|have)\b",
        normalized,
    ):
        return True
    return bool(re.search(r"^how did\b.*\bsuggest\b", normalized))


def asks_assistant_memory(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in features.ASSISTANT_MEMORY_REFERENCE_CUES)


def temporal_strategy() -> QuestionStrategy:
    return QuestionStrategy("temporal", True, True, True, False, True, "general", 80)


def multi_evidence_strategy() -> QuestionStrategy:
    return QuestionStrategy("multi_evidence", True, True, True, False, True, "general", 80)


def recency_strategy() -> QuestionStrategy:
    return QuestionStrategy("recency", True, True, True, False, True, "general", 60)


def preference_strategy() -> QuestionStrategy:
    return QuestionStrategy("preference", True, True, True, True, False, "preference", 60)


def factual_strategy() -> QuestionStrategy:
    return QuestionStrategy("factual", False, False, False, False, False, "general", 40)


def strategy_by_name(name: str) -> QuestionStrategy:
    if name == "temporal":
        return temporal_strategy()
    if name == "multi_evidence":
        return multi_evidence_strategy()
    if name == "recency":
        return recency_strategy()
    if name == "preference":
        return preference_strategy()
    return factual_strategy()


def is_preference_question(
    lowered: str,
    *,
    inference_signals: bool = False,
    inference_mode: str = "broad",
) -> bool:
    if any(cue in lowered for cue in features.ADVICE_REQUEST_CUES):
        return True
    if not inference_signals:
        return False
    return is_personalized_inference_question(lowered, mode=inference_mode)


def is_personalized_inference_question(lowered: str, *, mode: str = "broad") -> bool:
    normalized = " ".join(lowered.split())
    if uses_conservative_inference_frames(mode):
        if excludes_retail_context_for_inference(mode, normalized):
            return False
        if excludes_binary_affinity_choice(mode, normalized):
            return False
        if has_relocation_inference_frame(normalized):
            return True
        if has_social_likelihood_frame(normalized):
            return True
        if has_affinity_or_benefit_frame(normalized):
            return True
        if has_fit_quality_frame(normalized):
            return True
        if has_sensitivity_inference_frame(normalized):
            return True
        return False
    if re.search(r"\bwould(?:n't| not)?\b", normalized):
        return True
    if re.search(r"\b(is it likely|likely|might)\b", normalized):
        return True
    return any(phrase in normalized for phrase in features.PERSONALIZED_INFERENCE_PHRASES)


def uses_conservative_inference_frames(mode: str) -> bool:
    return mode in {
        "conservative_inference_basic",
        "conservative_inference_no_retail",
        "conservative_inference",
    }


def excludes_retail_context_for_inference(mode: str, normalized: str) -> bool:
    return mode in {"conservative_inference_no_retail", "conservative_inference"} and bool(
        re.search(r"\b(shop|store)\b", normalized)
    )


def excludes_binary_affinity_choice(mode: str, normalized: str) -> bool:
    return mode == "conservative_inference" and bool(re.search(r"\bwould\b.*\benjoy\b.*\bor\b", normalized))


def has_relocation_inference_frame(normalized: str) -> bool:
    return bool(
        re.search(
            r"\bwould\b.*\b(want to move|move back|open to moving|open to move|move to another country)\b",
            normalized,
        )
    )


def has_social_likelihood_frame(normalized: str) -> bool:
    return bool(re.search(r"\b(is it likely|likely)\b.*\b(friends?|social circle|teammates?)\b", normalized))


def has_affinity_or_benefit_frame(normalized: str) -> bool:
    return bool(re.search(r"\b(would|might|likely)\b.*\b(enjoy|like|benefit from)\b", normalized))


def has_fit_quality_frame(normalized: str) -> bool:
    return "good hobby" in normalized or "good fit" in normalized


def has_sensitivity_inference_frame(normalized: str) -> bool:
    return bool(re.search(r"\b(would|wouldn't|would not)\b.*\b(discomfort|allerg|trigger)\b", normalized))


def configured_profile(name: str, config: dict[str, Any] | None) -> str:
    if config and bool(config.get("retrieval", {}).get("multi_evidence_word_boundary", False)):
        return f"{name}_boundary"
    return name


def router_mode(config: dict[str, Any] | None) -> str:
    raw_mode = "legacy" if not config else str(config.get("retrieval", {}).get("router_mode", "legacy"))
    return normalize_router_mode(raw_mode)


def collection_intent_mode(config: dict[str, Any] | None) -> str:
    if not config:
        return "specific"
    retrieval_cfg = config.get("retrieval", {})
    return normalize_collection_intent_mode(str(retrieval_cfg.get("collection_intent_mode", "specific")))


def use_collection_intent_routing(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    retrieval_cfg = config.get("retrieval", {})
    return bool(retrieval_cfg.get("collection_intent_routing", False))


def use_personalized_inference_routing(config: dict[str, Any] | None) -> bool:
    if not config:
        return False
    retrieval_cfg = config.get("retrieval", {})
    return bool(retrieval_cfg.get("personalized_inference_routing", False))


def preference_inference_mode(config: dict[str, Any] | None) -> str:
    if not config:
        return "broad"
    retrieval_cfg = config.get("retrieval", {})
    return normalize_preference_inference_mode(str(retrieval_cfg.get("personalized_inference_mode", "broad")))


def is_multi_evidence_question(
    lowered: str,
    *,
    word_boundary: bool = False,
    collection_intent_routing: bool = False,
    collection_intent_mode: str = "specific",
) -> bool:
    if collection_intent_routing and is_collection_intent_question(lowered, mode=collection_intent_mode):
        return True
    if not word_boundary:
        return any(cue in lowered for cue in features.AGGREGATION_CUES)

    if any(cue in lowered for cue in features.AGGREGATION_PHRASES):
        return True
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in features.AGGREGATION_TERMS)


def is_collection_intent_question(lowered: str, *, mode: str = "specific") -> bool:
    normalized = " ".join(lowered.split())
    mode = normalize_collection_intent_mode(mode)
    if mode == "surface_head":
        return is_plural_enumerable_head_question(normalized)
    if mode == "bounded_head":
        return is_bounded_enumerable_head_question(normalized)
    if mode == "core_or_recurring":
        return is_core_set_or_name_list_question(normalized) or (
            is_bounded_enumerable_head_question(normalized) and has_enumerable_recurring_scope(normalized)
        )
    if mode == "semantic_enumerable":
        return is_semantic_enumerable_question(normalized)
    if mode == "explicit_or_temporal":
        return is_explicit_enumerable_question(normalized) or (
            is_bounded_enumerable_head_question(normalized) and has_enumerable_temporal_scope(normalized)
        )
    if mode == "explicit_or_recurring":
        return is_explicit_enumerable_question(normalized) or (
            is_bounded_enumerable_head_question(normalized) and has_enumerable_recurring_scope(normalized)
        )
    return is_explicit_enumerable_question(normalized)


def is_explicit_enumerable_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has)\b.*\bboth\b", normalized):
        return True
    if re.search(r"\b(what|which)\b.*\bboth\b", normalized):
        return True
    if normalized.startswith("what are the names of ") or normalized.startswith("what are ") and " names" in normalized:
        return True
    return starts_with_wh_entity_head(
        normalized,
        operator="what",
        heads=features.WHAT_ENUMERABLE_ENTITY_HEADS,
    ) or starts_with_wh_entity_head(
        normalized,
        operator="which",
        heads=features.WHICH_LOCATION_ENTITY_HEADS,
    )


def starts_with_wh_entity_head(normalized: str, *, operator: str, heads: tuple[str, ...]) -> bool:
    return any(normalized.startswith(f"{operator} {head}") for head in heads)


def is_core_set_or_name_list_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    return bool(re.search(r"^(?:what are|which)\b.*\bnames\b", normalized))


def is_semantic_enumerable_question(normalized: str) -> bool:
    if is_non_enumerable_attribute_question(normalized):
        return False
    if is_core_set_or_name_list_question(normalized):
        return True
    if is_bounded_enumerable_head_question(normalized) and has_enumerable_recurring_scope(normalized):
        return True
    if not normalized.startswith(("what ", "which ")):
        return False
    if is_descriptive_or_advice_question(normalized):
        return False

    match = re.search(r"^(?:what|which)\s+(.+?)\s+\b(?:has|have|had|does|do|did|is|are|was|were)\b", normalized)
    if not match:
        return False
    head_tokens = re.findall(r"[a-z0-9]+", match.group(1))
    if not has_plural_enumerable_head(head_tokens):
        return False
    return has_memory_experience_predicate(normalized)


def has_plural_enumerable_head(tokens: list[str]) -> bool:
    return any(
        token not in features.ENUMERABLE_HEAD_STOPWORDS
        and (token in features.IRREGULAR_ENUMERABLE_HEADS or (token.endswith("s") and len(token) > 3))
        for token in tokens
    )


def has_memory_experience_predicate(normalized: str) -> bool:
    return any(re.search(rf"\b{re.escape(predicate)}\b", normalized) for predicate in features.MEMORY_EVENT_PREDICATES)


def is_non_enumerable_question(normalized: str) -> bool:
    if any(cue in normalized for cue in features.NON_ENUMERABLE_SURFACE_CUES):
        return True
    if re.search(r"\bwhat does\b.*\bdo to\b", normalized):
        return True
    if re.search(r"\bwhat do\b.*\blike\b", normalized):
        return True
    return bool(re.search(r"\bwhat is\b.*\bstatus\b", normalized))


def is_non_enumerable_attribute_question(normalized: str) -> bool:
    if is_non_enumerable_question(normalized):
        return True
    if any(cue in normalized for cue in features.NON_ENUMERABLE_STYLE_CUES):
        return True
    return bool(re.search(r"^which\b.+\b[a-z]+'s\b", normalized))


def should_keep_auto_route(question: str, config: dict[str, Any] | None) -> bool:
    mode = auto_route_guard_mode(config)
    if mode == "off":
        return False
    if is_recommendation_recall_question(question) or is_targeted_compatibility_question(question):
        return True
    if mode == "perspective_v1":
        return not is_autobiographical_question(question)
    return False


def auto_route_guard_mode(config: dict[str, Any] | None) -> str:
    if not config:
        return "off"
    return str(config.get("retrieval", {}).get("auto_route_guard_mode", "off"))


def is_autobiographical_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    return bool(
        re.search(
            r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b",
            normalized,
        )
    )


def is_targeted_compatibility_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    return bool(
        re.search(r"\bwould(?:n't| not)?\b.*\b(?:cause|trigger|discomfort|allerg)", normalized)
        or re.search(r"\b(?:cause|trigger)\b.*\b(?:discomfort|allerg)", normalized)
    )


def is_plural_enumerable_head_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    if re.search(r"^(?:what are|which)\b.*\bnames\b", normalized):
        return True
    if not normalized.startswith(("what ", "which ")):
        return False
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) < 2:
        return False
    content = [token for token in tokens[1:8] if token not in features.PLURAL_HEAD_STOPWORDS]
    return any(token.endswith("s") and len(token) > 3 for token in content)


def is_bounded_enumerable_head_question(normalized: str) -> bool:
    """Identify explicit enumerable-list questions while avoiding descriptive attributes."""
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    if re.search(r"^(?:what are|which)\b.*\bnames\b", normalized):
        return True
    if not normalized.startswith(("what ", "which ")):
        return False
    if is_descriptive_or_advice_question(normalized):
        return False
    return is_plural_enumerable_head_question(normalized)


def is_descriptive_or_advice_question(normalized: str) -> bool:
    if normalized.startswith(features.DESCRIPTIVE_QUESTION_HEADS):
        return True
    return any(re.search(pattern, normalized) for pattern in features.DESCRIPTIVE_QUESTION_PATTERNS)


def is_generic_preference_question(
    lowered: str,
    *,
    inference_signals: bool = False,
    inference_mode: str = "broad",
) -> bool:
    if any(cue in lowered for cue in features.GENERIC_ADVICE_REQUEST_CUES):
        return True
    if not inference_signals:
        return False
    normalized = " ".join(lowered.split())
    if inference_mode == "contextual_inference":
        if re.search(r"\b(would|might|could)\b.*\b(pursue|want|consider|choose|do|have|be)\b", normalized):
            return True
        if " based on " in normalized and re.search(r"\b(would|might|could|likely)\b", normalized):
            return True
    if re.search(r"\b(would|might|likely|could)\b.*\b(enjoy|like|prefer|benefit|fit)\b", normalized):
        return True
    if re.search(r"\b(is it likely|likely)\b", normalized):
        return True
    return "good fit" in normalized or "good for me" in normalized


def is_generic_temporal_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.TEMPORAL_SURFACE_CUES)


def is_generic_recency_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.GENERIC_RECENCY_CUES)


def is_generic_order_route_question(lowered: str) -> bool:
    if "chronological" in lowered or "earliest to latest" in lowered or "order of" in lowered:
        return True
    return bool(re.search(r"\b(?:what|which).*\b(?:happened|occurred|came|took place)\s+first\b", lowered))


def is_generic_duration_route_question(lowered: str) -> bool:
    return bool(
        "how long" in lowered
        or "since" in lowered
        or re.search(r"\bhow many\s+(?:days|weeks|months|years)\b", lowered)
        or re.search(r"\b(?:days|weeks|months|years)\s+(?:ago|passed|between)\b", lowered)
    )


def is_generic_historical_state_question(lowered: str) -> bool:
    return any(cue in lowered for cue in ("previous", "initial", "original", "used to"))


def is_generic_most_recent_question(lowered: str) -> bool:
    return "most recent" in lowered or "most recently" in lowered or "latest" in lowered


def is_generic_temporal_comparison_quantity_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.TEMPORAL_QUANTITY_CUES) and any(
        cue in lowered for cue in features.TEMPORAL_ORDER_CUES
    )


def use_broad_enumerable_context_window_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if any(re.search(pattern, normalized) for pattern in features.BROAD_ENUMERABLE_CONTEXT_PATTERNS):
        return True
    if any(re.search(pattern, normalized) for pattern in features.EVENT_ENUMERABLE_CONTEXT_PATTERNS):
        return True
    return False


def use_temporally_scoped_enumerable_context_window_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if "most frequently" in normalized or "most frequent" in normalized:
        return False
    if any(re.search(pattern, normalized) for pattern in features.TEMPORALLY_SCOPED_ENUMERABLE_CONTEXT_PATTERNS):
        return True
    if any(re.search(pattern, normalized) for pattern in features.EVENT_ENUMERABLE_CONTEXT_PATTERNS) and (
        has_explicit_time_scope(normalized)
    ):
        return True
    return False


def has_explicit_time_scope(normalized_question: str) -> bool:
    if re.search(r"\b20\d{2}\b", normalized_question):
        return True
    return any(re.search(rf"\b{month}\b", normalized_question) for month in features.MONTH_NAMES)


def has_enumerable_temporal_scope(normalized_question: str) -> bool:
    if has_explicit_time_scope(normalized_question):
        return True
    if re.search(
        r"\b(?:past|last|next|this)\s+(?:day|week|month|year|summer|winter|spring|fall|autumn)\b",
        normalized_question,
    ):
        return True
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", normalized_question):
        return True
    if re.search(
        r"\b(?:mid|early|late)[-\s]+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
        normalized_question,
    ):
        return True
    return False


def has_enumerable_recurring_scope(normalized_question: str) -> bool:
    """Detect recurring or multi-anchor temporal scopes, not ordinary dated events."""
    if normalized_question.startswith(("what should ", "which should ")):
        return False
    if any(cue in normalized_question for cue in features.RECURRING_SCOPE_EXCLUSION_CUES):
        return False

    weekday = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    if len(re.findall(rf"\b{weekday}s?\b", normalized_question)) >= 2:
        return True
    if re.search(rf"\bon\s+{weekday}s\b", normalized_question):
        return True
    if re.search(rf"\b(?:every|each)\s+{weekday}s?\b", normalized_question):
        return True
    if re.search(r"\b(?:on|every|each)\s+weekends?\b", normalized_question):
        return True
    if re.search(r"\b(?:weekdays|weekends)\b", normalized_question) and re.search(
        r"\b(?:usually|typically|regularly|routinely)\b", normalized_question
    ):
        return True
    if re.search(
        r"\b(?:daily|weekly|monthly|yearly|annually|regularly|routinely|usually|typically)\b",
        normalized_question,
    ):
        return True
    if re.search(r"\b(?:every|each)\s+(?:day|week|month|year|morning|evening|night)\b", normalized_question):
        return True
    return False


def is_temporal_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.TEMPORAL_RETRIEVAL_CUES)


def is_strict_temporal_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.STRICT_TEMPORAL_CUES) or any(
        re.search(pattern, lowered) for pattern in features.STRICT_TEMPORAL_PATTERNS
    )


def is_recency_question(lowered: str) -> bool:
    return any(cue in lowered for cue in features.RECENCY_STATE_CUES)


def is_order_route_question(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in features.ORDERING_CUES)


def is_duration_route_question(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in features.DURATION_CUES)


def use_duration_evidence_requirements_question(question: str, *, mode: str = "all") -> bool:
    if mode == "all":
        return is_duration_route_question(question)
    if mode not in {"duration_shape", "duration_shape_without_elapsed_since", "strict_duration_shape"}:
        return False
    normalized = " ".join(question.lower().split())
    if mode == "strict_duration_shape":
        if re.search(r"\bhow long\b.*\btake\b", normalized):
            return True
        if re.search(r"\bhow many\s+days\s+did it take\b", normalized):
            return True
        if re.search(r"\bhow many\s+days\s+(?:had\s+)?passed between\b", normalized):
            return True
        if re.search(r"\bdays\s+passed between\b", normalized):
            return True
        return False
    if mode == "duration_shape_without_elapsed_since" and "how long has it been since" in normalized:
        return False
    if "how long" in normalized or "for how long" in normalized:
        return True
    if re.search(r"\bhow many\s+(?:days|weeks|months|years)\s+did it take\b", normalized):
        return True
    if re.search(r"\bhow many\s+(?:days|weeks|months|years)\s+(?:had\s+)?passed between\b", normalized):
        return True
    if re.search(r"\b(?:days|weeks|months|years)\s+passed between\b", normalized):
        return True
    return False


def is_historical_state_question(question: str) -> bool:
    if asks_assistant_memory(question):
        return False
    lowered = question.lower()
    return any(cue in lowered for cue in features.HISTORICAL_STATE_CUES)


def is_most_recent_question(question: str) -> bool:
    lowered = question.lower()
    return "most recent" in lowered or "most recently" in lowered


def is_temporal_comparison_quantity_question(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in features.TEMPORAL_QUANTITY_CUES) and "earlier" in lowered

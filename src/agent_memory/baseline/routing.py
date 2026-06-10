from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from agent_memory.baseline.config_aliases import (
    normalize_collection_intent_mode,
    normalize_preference_inference_mode,
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
    if router_mode(config) == "generic_v1":
        return choose_memory_route_generic(example, config)

    question = example.question
    route = "auto"
    collection_intent = use_collection_intent_routing(config) and is_collection_intent_question(
        question.lower(),
        mode=collection_intent_mode(config),
    )
    base_strategy = choose_strategy(
        question,
        profile=configured_profile("legacy_auto", config),
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
        and not collection_intent
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
    collection_intent = use_collection_intent_routing(config) and is_collection_intent_question(
        lowered,
        mode=collection_intent_mode(config),
    )
    base_strategy = choose_strategy(
        question,
        profile=configured_profile("legacy_auto", config),
        collection_intent_routing=use_collection_intent_routing(config),
        personalized_inference_routing=use_personalized_inference_routing(config),
        preference_inference_mode=preference_inference_mode(config),
        routing_mode="generic_v1",
        collection_intent_mode=collection_intent_mode(config),
    )
    if (
        base_strategy.name in {"multi_evidence", "preference"}
        and not asks_assistant_memory(question)
        and not is_generic_temporal_comparison_quantity_question(lowered)
        and not collection_intent
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
            strategy_profile=profile("legacy_auto"),
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
        strategy_profile=profile("legacy_auto"),
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
    routing_mode: str = "legacy",
    collection_intent_mode: str = "specific",
) -> QuestionStrategy:
    lowered = question.lower()
    use_word_boundary = profile.endswith("_boundary")
    base_profile = profile.removesuffix("_boundary")
    if routing_mode == "generic_v1":
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
    if base_profile == "legacy_auto":
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
    if "how many" in lowered and ("replace or fix" in lowered or "replaced or fixed" in lowered):
        return True
    if "years older" in lowered or "year older" in lowered or "how much older" in lowered:
        return True
    return "total number of siblings" in lowered or "how many siblings" in lowered


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
    signals = (
        "previous chat",
        "previous conversation",
        "previously discussed",
        "looking back",
        "trying to recall",
        "we discussed",
        "we talked about",
        "talked about last time",
        "last time",
        "do you remember",
        "remind me",
        "you suggested",
        "you said",
        "you told",
        "you mentioned",
        "you provided",
        "you gave",
        "you recommended",
        "you outlined",
        "you created",
        "you wrote",
        "you called",
        "you advised",
        "what did you say",
        "what did you tell",
        "what did you suggest",
        "what did you recommend",
        "what advice did you",
        "your suggestion",
        "your recommendation",
    )
    return any(signal in lowered for signal in signals)


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
    signals = (
        "recommend",
        "suggest",
        "advice",
        "any tips",
        "what do you think",
        "should i",
        "would it be a good idea",
        "help me decide",
        "which should i",
        "what should i",
        "do you think it",
    )
    if any(signal in lowered for signal in signals):
        return True
    if not inference_signals:
        return False
    return is_personalized_inference_question(lowered, mode=inference_mode)


def is_personalized_inference_question(lowered: str, *, mode: str = "broad") -> bool:
    normalized = " ".join(lowered.split())
    if mode in {"narrow", "narrow_v2", "narrow_v3"}:
        if mode in {"narrow_v2", "narrow_v3"} and re.search(r"\b(shop|store)\b", normalized):
            return False
        if mode == "narrow_v3" and re.search(r"\bwould\b.*\benjoy\b.*\bor\b", normalized):
            return False
        if re.search(r"\bwould\b.*\b(want to move|move back|open to moving|open to move|move to another country)\b", normalized):
            return True
        if re.search(r"\b(is it likely|likely)\b.*\b(friends?|social circle|teammates?)\b", normalized):
            return True
        if re.search(r"\b(would|might|likely)\b.*\b(enjoy|like|benefit from)\b", normalized):
            return True
        if "good hobby" in normalized or "good fit" in normalized:
            return True
        if re.search(r"\b(would|wouldn't|would not)\b.*\b(discomfort|allerg|trigger)\b", normalized):
            return True
        return False
    if re.search(r"\bwould(?:n't| not)?\b", normalized):
        return True
    if re.search(r"\b(is it likely|likely|might)\b", normalized):
        return True
    inferential_phrases = (
        "benefit from",
        "good fit",
        "be good for",
        "would enjoy",
        "might enjoy",
        "likely enjoy",
        "would like",
        "might like",
        "likely like",
        "might consider",
        "could consider",
        "alternative career",
        "underlying condition",
    )
    return any(phrase in normalized for phrase in inferential_phrases)


def configured_profile(name: str, config: dict[str, Any] | None) -> str:
    if config and bool(config.get("retrieval", {}).get("multi_evidence_word_boundary", False)):
        return f"{name}_boundary"
    return name


def router_mode(config: dict[str, Any] | None) -> str:
    if not config:
        return "legacy"
    return str(config.get("retrieval", {}).get("router_mode", "legacy"))


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
        signals = (
            "how many",
            "how much",
            "total",
            "in total",
            "combined",
            "average",
            "percentage",
            "difference",
            "more than",
            "less than",
            "spent",
            "cost",
            "count",
            "sum",
        )
        return any(signal in lowered for signal in signals)

    phrase_signals = (
        "how many",
        "how much",
        "in total",
        "combined",
        "average",
        "percentage",
        "difference",
        "more than",
        "less than",
    )
    if any(signal in lowered for signal in phrase_signals):
        return True
    word_signals = ("total", "spent", "cost", "costs", "count", "counts", "sum")
    return any(re.search(rf"\b{re.escape(signal)}\b", lowered) for signal in word_signals)


def is_collection_intent_question(lowered: str, *, mode: str = "specific") -> bool:
    normalized = " ".join(lowered.split())
    if mode == "generic_v1":
        return is_generic_list_collection_question(normalized)
    if mode == "generic_v2":
        return is_generic_list_collection_question_v2(normalized)
    if mode == "core_plus_periodic_generic_v1":
        return is_core_set_or_name_list_question(normalized) or (
            is_generic_list_collection_question_v2(normalized) and has_generic_list_periodic_scope(normalized)
        )
    if mode == "semantic_enumerable_v2":
        return is_semantic_enumerable_list_question_v2(normalized)
    if mode == "specific_plus_temporal_generic_v1":
        return is_specific_list_collection_question(normalized) or (
            is_generic_list_collection_question_v2(normalized) and has_generic_list_temporal_scope(normalized)
        )
    if mode == "specific_plus_periodic_generic_v1":
        return is_specific_list_collection_question(normalized) or (
            is_generic_list_collection_question_v2(normalized) and has_generic_list_periodic_scope(normalized)
        )
    return is_specific_list_collection_question(normalized)

def is_specific_list_collection_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has)\b.*\bboth\b", normalized):
        return True
    if re.search(r"\b(what|which)\b.*\bboth\b", normalized):
        return True
    if normalized.startswith("what are the names of ") or normalized.startswith("what are ") and " names" in normalized:
        return True
    collection_signals = (
        "what books",
        "what activities",
        "what items",
        "what people",
        "what shelters",
        "what states",
        "what countries",
        "what european countries",
        "which countries",
        "which european countries",
        "which us cities",
        "which u.s. cities",
        "which city",
        "which cities",
        "which geographical locations",
        "what authors",
        "what pets",
        "what damages",
        "what subjects",
        "what kind of interests",
    )
    return any(normalized.startswith(signal) for signal in collection_signals)


def is_core_set_or_name_list_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    return bool(re.search(r"^(?:what are|which)\b.*\bnames\b", normalized))


def is_semantic_enumerable_list_question_v2(normalized: str) -> bool:
    if is_non_enumerable_question_v2(normalized):
        return False
    if is_core_set_or_name_list_question(normalized):
        return True
    if is_generic_list_collection_question_v2(normalized) and has_generic_list_periodic_scope(normalized):
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
    stop = {
        "what",
        "which",
        "are",
        "were",
        "is",
        "was",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "of",
        "my",
        "your",
    }
    irregular_plural = {"people", "children", "kids"}
    return any(
        token not in stop and (token in irregular_plural or (token.endswith("s") and len(token) > 3))
        for token in tokens
    )


def has_memory_experience_predicate(normalized: str) -> bool:
    predicate_signals = (
        "read",
        "visit",
        "visited",
        "been to",
        "buy",
        "bought",
        "made",
        "done",
        "partake",
        "participated",
        "attended",
        "met",
        "helped",
        "mention",
        "mentioned",
        "volunteer",
        "vacationed",
        "collect",
        "collected",
        "enjoy",
        "have",
        "has",
        "own",
        "owns",
        "adopted",
        "recommend",
        "recommended",
        "use",
        "uses",
        "used",
        "watch",
        "watched",
        "seen",
        "travel",
        "traveled",
        "travelled",
        "pursue",
        "pursued",
        "happened",
    )
    return any(re.search(rf"\b{re.escape(signal)}\b", normalized) for signal in predicate_signals)


def is_non_enumerable_question(normalized: str) -> bool:
    phrase_signals = (
        "would",
        "wouldn't",
        "would not",
        "might",
        "likely",
        "relationship status",
        "what are some ",
        "what personality traits",
        "what kind of",
        "what type of",
        "which type of",
        "what are the skills",
        "what similar",
        "what things",
        "what causes",
        "what areas",
    )
    if any(signal in normalized for signal in phrase_signals):
        return True
    if re.search(r"\bwhat does\b.*\bdo to\b", normalized):
        return True
    if re.search(r"\bwhat do\b.*\blike\b", normalized):
        return True
    return bool(re.search(r"\bwhat is\b.*\bstatus\b", normalized))


def is_non_enumerable_question_v2(normalized: str) -> bool:
    if is_non_enumerable_question(normalized):
        return True
    phrase_signals = (
        "total number",
        "what style",
        "what styles",
        "what kind of style",
        "which style",
        "which styles",
    )
    if any(signal in normalized for signal in phrase_signals):
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


def is_generic_list_collection_question(normalized: str) -> bool:
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    if re.search(r"^(?:what are|which)\b.*\bnames\b", normalized):
        return True
    if not normalized.startswith(("what ", "which ")):
        return False
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if len(tokens) < 2:
        return False
    stop = {
        "what",
        "which",
        "are",
        "were",
        "is",
        "was",
        "do",
        "did",
        "does",
        "have",
        "has",
        "the",
        "a",
        "an",
        "my",
        "i",
        "we",
        "you",
        "first",
        "last",
        "latest",
        "current",
        "previous",
        "initial",
        "original",
        "date",
        "time",
        "year",
        "month",
        "day",
    }
    content = [token for token in tokens[1:8] if token not in stop]
    return any(token.endswith("s") and len(token) > 3 for token in content)


def is_generic_list_collection_question_v2(normalized: str) -> bool:
    """Identify explicit enumerable-list questions while avoiding descriptive attributes."""
    if re.search(r"\b(did|do|does|have|has|what|which)\b.*\bboth\b", normalized):
        return True
    if re.search(r"^(?:what are|which)\b.*\bnames\b", normalized):
        return True
    if not normalized.startswith(("what ", "which ")):
        return False
    if is_descriptive_or_advice_question(normalized):
        return False
    return is_generic_list_collection_question(normalized)


def is_descriptive_or_advice_question(normalized: str) -> bool:
    descriptive_heads = (
        "what emotions",
        "what feelings",
        "what attributes",
        "what traits",
        "what qualities",
        "what characteristics",
        "what habits",
        "what dreams",
        "what goals",
        "what challenges",
        "what difficulties",
        "what problems",
        "what ways",
        "what methods",
        "what reasons",
        "what thoughts",
        "what opinions",
        "what memories",
        "what progress",
        "what focus",
        "what kind of dream",
    )
    if normalized.startswith(descriptive_heads):
        return True
    descriptive_patterns = (
        r"\bwhat\b.*\b(?:represent|describe|mean|symbolize)\b",
        r"\bwhat\b.*\b(?:motivat(?:e|es|ed|ing)|inspir(?:e|es|ed|ing))\b",
        r"\bwhat\b.*\b(?:think|feel|say)\b.*\babout\b",
        r"\bwhat\b.*\b(?:recommend|recommended|recommendations?|suggest|suggested|suggestions?|advice|tips)\b",
        r"\bwhat\b.*\b(?:use|uses|used)\b.*\bfor\b",
    )
    return any(re.search(pattern, normalized) for pattern in descriptive_patterns)


def is_generic_preference_question(
    lowered: str,
    *,
    inference_signals: bool = False,
    inference_mode: str = "broad",
) -> bool:
    signals = (
        "recommend",
        "suggest",
        "advice",
        "tips",
        "should i",
        "would it be a good idea",
        "help me decide",
        "which should i",
        "what should i",
    )
    if any(signal in lowered for signal in signals):
        return True
    if not inference_signals:
        return False
    normalized = " ".join(lowered.split())
    if inference_mode == "generic_v2":
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
    signals = (
        "when",
        "what date",
        "what time",
        "how long",
        "duration",
        "since",
        "before",
        "after",
        "earlier",
        "later",
        "first",
        "chronological",
        "earliest",
        "latest",
    )
    return any(signal in lowered for signal in signals)


def is_generic_recency_question(lowered: str) -> bool:
    signals = ("current", "currently", "now", "latest", "most recent", "recently", "previous", "initial", "original", "used to")
    return any(signal in lowered for signal in signals)


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
    return any(signal in lowered for signal in ("previous", "initial", "original", "used to"))


def is_generic_most_recent_question(lowered: str) -> bool:
    return "most recent" in lowered or "most recently" in lowered or "latest" in lowered


def is_generic_temporal_comparison_quantity_question(lowered: str) -> bool:
    quantity_signals = ("how many", "how much", "difference", "compared")
    return any(signal in lowered for signal in quantity_signals) and any(signal in lowered for signal in ("earlier", "later", "before", "after"))


def use_list_context_window_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if re.search(r"\b(which|what)\s+(?:u\.?s\.?\s+)?cit(?:y|ies)\b", normalized):
        return True
    if re.search(r"\b(which|what)\s+(?:european\s+)?countr(?:y|ies)\b", normalized):
        return True
    if "which geographical locations" in normalized:
        return True
    if re.search(r"\bwhat subject\b.*\bboth\b", normalized):
        return True
    if re.search(r"\bwhat do\b.*\bboth have in common\b", normalized):
        return True
    if re.search(r"\bwhat items\b.*\b(buy|bought|purchase|purchased)\b", normalized):
        return True
    return False


def use_list_context_window_question_v2(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    if "most frequently" in normalized or "most frequent" in normalized:
        return False
    if re.search(r"\b(which|what)\s+(?:u\.?s\.?\s+)?cities\b", normalized):
        return True
    if re.search(r"\b(which|what)\s+(?:european\s+)?countries\b", normalized):
        return True
    if "which geographical locations" in normalized:
        return True
    if re.search(r"\bwhat subject\b.*\bboth\b", normalized):
        return True
    if re.search(r"\bwhat do\b.*\bboth have in common\b", normalized):
        return True
    if re.search(r"\bwhat items\b.*\b(buy|bought|purchase|purchased)\b", normalized) and has_explicit_time_scope(
        normalized
    ):
        return True
    return False


def has_explicit_time_scope(normalized_question: str) -> bool:
    if re.search(r"\b20\d{2}\b", normalized_question):
        return True
    months = (
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    )
    return any(re.search(rf"\b{month}\b", normalized_question) for month in months)


def has_generic_list_temporal_scope(normalized_question: str) -> bool:
    if has_explicit_time_scope(normalized_question):
        return True
    if re.search(r"\b(?:past|last|next|this)\s+(?:day|week|month|year|summer|winter|spring|fall|autumn)\b", normalized_question):
        return True
    if re.search(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", normalized_question):
        return True
    if re.search(r"\b(?:mid|early|late)[-\s]+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b", normalized_question):
        return True
    return False


def has_generic_list_periodic_scope(normalized_question: str) -> bool:
    """Detect recurring or multi-anchor temporal scopes, not ordinary dated events."""
    if normalized_question.startswith(("what should ", "which should ")):
        return False
    quantity_signals = ("how many", "how much", "total", "average", "sum", "combined", "difference")
    if any(signal in normalized_question for signal in quantity_signals):
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
    if re.search(r"\b(?:daily|weekly|monthly|yearly|annually|regularly|routinely|usually|typically)\b", normalized_question):
        return True
    if re.search(r"\b(?:every|each)\s+(?:day|week|month|year|morning|evening|night)\b", normalized_question):
        return True
    return False


def is_temporal_question(lowered: str) -> bool:
    signals = (
        "when",
        "what date",
        "which happened first",
        "what happened first",
        "happened first",
        "occurred first",
        "meet first",
        "met first",
        "order of",
        "from earliest",
        "earliest to latest",
        "chronological",
        "ago",
        "before",
        "after",
        "earlier",
        "later",
        "how long",
        "duration",
        "since",
        "last month",
        "past month",
        "past two weeks",
        "this year",
    )
    return any(signal in lowered for signal in signals)


def is_strict_temporal_question(lowered: str) -> bool:
    signals = (
        "when",
        "what date",
        "what time",
        "how long",
        "how old",
        "order of",
        "from earliest",
        "earliest to latest",
        "chronological",
        "which happened first",
        "what happened first",
        "happened first",
        "occurred first",
        "meet first",
        "met first",
        "first issue",
        "last month",
        "past month",
        "previous frequent flyer",
        "previous goal",
        "before i started",
        "before today",
        "move you made after",
        "period after",
    )
    return any(signal in lowered for signal in signals)


def is_recency_question(lowered: str) -> bool:
    signals = (
        "current",
        "currently",
        "now",
        "latest",
        "most recent",
        "recent",
        "previous",
        "previously",
        "initial",
        "initially",
        "original",
        "usually",
        "used to",
    )
    return any(signal in lowered for signal in signals)


def is_order_route_question(question: str) -> bool:
    lowered = question.lower()
    signals = (
        "order of",
        "from earliest to latest",
        "earliest to latest",
        "chronological",
        "which happened first",
        "what happened first",
        "happened first",
        "occurred first",
        "meet first",
        "met first",
        "purchase first",
        "purchased first",
        "buy first",
        "bought first",
        "first,",
    )
    return any(signal in lowered for signal in signals)


def is_duration_route_question(question: str) -> bool:
    lowered = question.lower()
    signals = (
        "how many days",
        "how many weeks",
        "how many months",
        "how long",
        "days ago",
        "weeks ago",
        "months ago",
        "since",
    )
    return any(signal in lowered for signal in signals)


def use_duration_evidence_requirements_question(question: str, *, mode: str = "all") -> bool:
    if mode == "all":
        return is_duration_route_question(question)
    if mode not in {"narrow_v1", "narrow_v2", "narrow_v3"}:
        return False
    normalized = " ".join(question.lower().split())
    if mode == "narrow_v3":
        if re.search(r"\bhow long\b.*\btake\b", normalized):
            return True
        if re.search(r"\bhow many\s+days\s+did it take\b", normalized):
            return True
        if re.search(r"\bhow many\s+days\s+(?:had\s+)?passed between\b", normalized):
            return True
        if re.search(r"\bdays\s+passed between\b", normalized):
            return True
        return False
    if mode == "narrow_v2" and "how long has it been since" in normalized:
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
    signals = ("previous", "initial", "initially", "original", "used to", "usually")
    return any(signal in lowered for signal in signals)


def is_most_recent_question(question: str) -> bool:
    lowered = question.lower()
    return "most recent" in lowered or "most recently" in lowered


def is_temporal_comparison_quantity_question(question: str) -> bool:
    lowered = question.lower()
    quantity_signals = ("how many", "how much", "difference", "compared")
    return any(signal in lowered for signal in quantity_signals) and "earlier" in lowered

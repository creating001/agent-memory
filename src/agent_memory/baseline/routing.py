from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_memory.core.schema import Example


METHOD_NAME = "v50_narrow_temporal_recency_verify"


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


def choose_lts_route(example: Example) -> str:
    question = example.question
    route = "auto"
    base_strategy = choose_strategy(question, profile="legacy_auto")
    if (
        base_strategy.name in {"multi_evidence", "preference"}
        and not asks_assistant_memory(question)
        and not is_temporal_comparison_quantity_question(question)
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


def route_settings(route: str, config: dict[str, Any]) -> RouteSettings:
    default_top_k = int(config["retrieval"].get("top_k", 40))
    if route == "prefer_user":
        return RouteSettings(
            strategy_profile="legacy_auto",
            top_k=40,
            prefer_user_chunks=True,
            use_evidence_rerank=False,
        )
    if route == "duration_temporal":
        return RouteSettings(
            strategy_profile="balanced",
            top_k=0,
            strategy_top_k={
                "factual": 40,
                "multi_evidence": 20,
                "preference": 40,
                "recency": 40,
                "temporal": 40,
            },
        )
    if route == "order_reflective":
        return RouteSettings(
            strategy_profile="temporal_first",
            top_k=40,
            use_reflective_retrieval=True,
            use_evidence_table=False,
            use_verification=True,
            verification_strategies=frozenset({"factual", "temporal"}),
        )
    if route == "state_history":
        return RouteSettings(strategy_profile="temporal_first", top_k=0)
    if route == "recency_auto":
        return RouteSettings(strategy_profile="temporal_first", top_k=40)
    return RouteSettings(strategy_profile="legacy_auto", top_k=default_top_k)


def choose_strategy(question: str, *, profile: str) -> QuestionStrategy:
    lowered = question.lower()
    if is_preference_question(lowered):
        return QuestionStrategy("preference", True, True, True, True, False, "preference", 60)
    if profile == "legacy_auto":
        if is_multi_evidence_question(lowered):
            return multi_evidence_strategy()
        if is_strict_temporal_question(lowered):
            return temporal_strategy()
        return factual_strategy()
    if profile == "balanced":
        if is_multi_evidence_question(lowered):
            return multi_evidence_strategy()
        if is_recency_question(lowered):
            return recency_strategy()
        if is_strict_temporal_question(lowered):
            return temporal_strategy()
        return factual_strategy()
    if is_temporal_question(lowered):
        return temporal_strategy()
    if is_multi_evidence_question(lowered):
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


def asks_assistant_memory(question: str) -> bool:
    lowered = question.lower()
    cues = (
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
    return any(cue in lowered for cue in cues)


def temporal_strategy() -> QuestionStrategy:
    return QuestionStrategy("temporal", True, True, True, False, True, "general", 80)


def multi_evidence_strategy() -> QuestionStrategy:
    return QuestionStrategy("multi_evidence", True, True, True, False, True, "general", 80)


def recency_strategy() -> QuestionStrategy:
    return QuestionStrategy("recency", True, True, True, False, True, "general", 60)


def factual_strategy() -> QuestionStrategy:
    return QuestionStrategy("factual", False, False, False, False, False, "general", 40)


def is_preference_question(lowered: str) -> bool:
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_multi_evidence_question(lowered: str) -> bool:
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_temporal_question(lowered: str) -> bool:
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_strict_temporal_question(lowered: str) -> bool:
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_recency_question(lowered: str) -> bool:
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_order_route_question(question: str) -> bool:
    lowered = question.lower()
    cues = (
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
    return any(cue in lowered for cue in cues)


def is_duration_route_question(question: str) -> bool:
    lowered = question.lower()
    cues = (
        "how many days",
        "how many weeks",
        "how many months",
        "how long",
        "days ago",
        "weeks ago",
        "months ago",
        "since",
    )
    return any(cue in lowered for cue in cues)


def is_historical_state_question(question: str) -> bool:
    if asks_assistant_memory(question):
        return False
    lowered = question.lower()
    cues = ("previous", "initial", "initially", "original", "used to", "usually")
    return any(cue in lowered for cue in cues)


def is_most_recent_question(question: str) -> bool:
    lowered = question.lower()
    return "most recent" in lowered or "most recently" in lowered


def is_temporal_comparison_quantity_question(question: str) -> bool:
    lowered = question.lower()
    quantity_cues = ("how many", "how much", "difference", "compared")
    return any(cue in lowered for cue in quantity_cues) and "earlier" in lowered

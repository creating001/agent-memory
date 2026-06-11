"""Stable public names for baseline configuration knobs.

The project keeps historical experiment identifiers for reproducibility, but
the default configs use descriptive names that are easier to explain in papers,
reports, and production handoff.  Centralizing aliases here keeps that boundary
explicit and avoids scattering experiment names through the pipeline.
"""

from __future__ import annotations


ROUTER_MODE_ALIASES = {
    "legacy": "rule_based",
    "generic_v1": "surface_intent",
    "llm_task_v1": "semantic_task",
    "llm_task_selective_v1": "selective_semantic_task",
}

STRATEGY_PROFILE_ALIASES = {
    "legacy_auto": "general",
}

COLLECTION_INTENT_MODE_ALIASES = {
    "enumerable_intent": "explicit_or_recurring",
    "core_enumerable_intent": "core_or_recurring",
    "semantic_enumerable": "semantic_enumerable",
    "generic_v1": "surface_head",
    "generic_v2": "bounded_head",
    "core_plus_periodic_generic_v1": "core_or_recurring",
    "semantic_enumerable_v2": "semantic_enumerable",
    "specific_plus_temporal_generic_v1": "explicit_or_temporal",
    "specific_plus_periodic_generic_v1": "explicit_or_recurring",
}

PREFERENCE_INFERENCE_MODE_ALIASES = {
    "personalized_inference": "conservative_inference",
    "conservative": "conservative_inference",
    "narrow": "conservative_inference_basic",
    "narrow_v2": "conservative_inference_no_retail",
    "narrow_v3": "conservative_inference",
    "generic_v2": "contextual_inference",
}

DURATION_EVIDENCE_MODE_ALIASES = {
    "duration_reasoning": "strict_duration_shape",
    "duration_questions": "strict_duration_shape",
    "conservative_duration": "strict_duration_shape",
    "narrow_v1": "duration_shape",
    "narrow_v2": "duration_shape_without_elapsed_since",
    "narrow_v3": "strict_duration_shape",
}

RERANK_GATE_ALIASES = {
    "factual_slot": "factual_action_slots",
    "factual_or_multi_evidence": "factual_action_slots_or_multi_evidence",
    "past_factual_slot": "past_or_singular_action_slots",
    "past_factual_or_temporal": "past_or_singular_action_slots_or_temporal",
    "simple_factual_slots_v1": "factual_slots",
    "simple_factual_slots_v2": "factual_action_slots",
    "simple_factual_slots_v2_or_multi_evidence": "factual_action_slots_or_multi_evidence",
    "simple_factual_slots_v3": "past_or_singular_action_slots",
    "simple_factual_slots_v3_or_temporal": "past_or_singular_action_slots_or_temporal",
}

MULTI_EVIDENCE_PROMPT_ALIASES = {
    "set_operation_aware": "set_operation_selective",
    "set_operations_when_needed": "set_operation_selective",
    "set_ops_v1_selective_v2": "set_operation_selective",
}

CONTEXT_WINDOW_MODE_ALIASES = {
    "enumerable_context": "temporal_enumerable_context",
    "list_context_v1": "broad_enumerable_context",
    "list_context_v2": "temporal_enumerable_context",
    "enumerable_context_v1": "broad_enumerable_context",
    "enumerable_context_v2": "temporal_enumerable_context",
}


def normalize_router_mode(mode: str) -> str:
    return ROUTER_MODE_ALIASES.get(mode, mode)


def normalize_strategy_profile(profile: str) -> str:
    return STRATEGY_PROFILE_ALIASES.get(profile, profile)


def normalize_collection_intent_mode(mode: str) -> str:
    return COLLECTION_INTENT_MODE_ALIASES.get(mode, mode)


def normalize_preference_inference_mode(mode: str) -> str:
    return PREFERENCE_INFERENCE_MODE_ALIASES.get(mode, mode)


def normalize_duration_evidence_mode(mode: str) -> str:
    return DURATION_EVIDENCE_MODE_ALIASES.get(mode, mode)


def normalize_rerank_gate(gate: str) -> str:
    return RERANK_GATE_ALIASES.get(gate, gate)


def normalize_multi_evidence_prompt_mode(mode: str) -> str:
    return MULTI_EVIDENCE_PROMPT_ALIASES.get(mode, mode)


def normalize_context_window_mode(mode: str) -> str:
    return CONTEXT_WINDOW_MODE_ALIASES.get(mode, mode)

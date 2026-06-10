"""Stable public names for baseline configuration knobs.

The project keeps historical experiment identifiers for reproducibility, but
the default configs use descriptive names that are easier to explain in papers,
reports, and production handoff.  Centralizing aliases here keeps that boundary
explicit and avoids scattering experiment names through the pipeline.
"""

from __future__ import annotations


COLLECTION_INTENT_MODE_ALIASES = {
    "enumerable_intent": "specific_plus_periodic_generic_v1",
    "core_enumerable_intent": "core_plus_periodic_generic_v1",
    "semantic_enumerable": "semantic_enumerable_v2",
}

PREFERENCE_INFERENCE_MODE_ALIASES = {
    "personalized_inference": "narrow_v3",
    "conservative": "narrow_v3",
}

DURATION_EVIDENCE_MODE_ALIASES = {
    "duration_questions": "narrow_v3",
    "conservative_duration": "narrow_v3",
}

RERANK_GATE_ALIASES = {
    "factual_slot": "simple_factual_slots_v2",
    "factual_or_multi_evidence": "simple_factual_slots_v2_or_multi_evidence",
    "past_factual_slot": "simple_factual_slots_v3",
    "past_factual_or_temporal": "simple_factual_slots_v3_or_temporal",
}

MULTI_EVIDENCE_PROMPT_ALIASES = {
    "set_operations_when_needed": "set_operation_selective",
    "set_ops_v1_selective_v2": "set_operation_selective",
}


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

from __future__ import annotations

from typing import Any

from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example


ROUTER_SYSTEM = """You classify memory QA questions for a general retrieval-and-answer pipeline.

Use only the question text and question date. Do not infer dataset, benchmark, category, sample id, or gold answer.
Do not answer the question. Return only JSON.
"""


ROUTER_TEMPLATE = """Question date: {question_date}
Question: {question}

Classify the question into one general task:
- single_fact: one directly supported fact or a small local fact is enough.
- multi_evidence: the answer requires collecting, counting, summing, comparing, or combining multiple distinct memories.
- preference_advice: the user asks for a recommendation, compatibility judgment, or likely preference based on prior personal evidence.
- temporal: the question asks for date/time, elapsed duration, chronology, or order.
- recency_state: the question asks for current, latest, most recent, previous, initial, or changed state.

Also classify the main operation:
- none, count, sum, compare, list, set_relation, duration, order, date_time, preference, recency.

Classify the requested answer slot:
- fact, person, place, time, duration, count, list, event, preference, state, other.

{target_support_instruction}

{temporal_hint_instruction}

Output JSON with this exact schema:
{{
  "task": "single_fact | multi_evidence | preference_advice | temporal | recency_state",
  "operation": "none | count | sum | compare | list | set_relation | duration | order | date_time | preference | recency",
  "temporal_subtype": "none | duration | order | date_time",
  "answer_slot": "fact | person | place | time | duration | count | list | event | preference | state | other",
{target_support_schema}
{temporal_hint_schema}
  "confidence": 0.0,
  "rationale": "short general reason"
}}

Return ONLY the JSON.
"""


TASKS = {"single_fact", "multi_evidence", "preference_advice", "temporal", "recency_state"}
OPERATIONS = {
    "none",
    "count",
    "sum",
    "compare",
    "list",
    "set_relation",
    "duration",
    "order",
    "date_time",
    "preference",
    "recency",
}
TEMPORAL_SUBTYPES = {"none", "duration", "order", "date_time"}
ANSWER_SLOTS = {"fact", "person", "place", "time", "duration", "count", "list", "event", "preference", "state", "other"}


def route_classification_messages(
    example: Example,
    *,
    include_temporal_hints: bool = False,
    include_target_support: bool = False,
) -> list[dict[str, str]]:
    temporal_hint_instruction = ""
    temporal_hint_schema = ""
    if include_temporal_hints:
        temporal_hint_instruction = (
            "If the question contains an explicit temporal expression that can help retrieve memories\n"
            "(relative, named, or date-range wording), resolve it against the question date when possible.\n"
            "These hints are for retrieval only; do not answer the question.\n"
        )
        temporal_hint_schema = (
            '  "temporal_search_hints": ["short resolved date/range/search phrase, or empty list"],\n'
        )
    target_support_instruction = ""
    target_support_schema = ""
    if include_target_support:
        target_support_instruction = (
            "Set target_support_required to true when a correct answer must match explicit target constraints from "
            "the question\n(for example a named person/entity, compared alternatives, a specified object/action/"
            "relation, or a scoped time range),\nso a related but mismatched memory would be unsafe.\n"
            "For required_target_phrases, copy short concrete target phrases from the question that must be directly "
            "supported by\nmemory evidence. Include names, compared alternatives, specific objects, scoped entities, "
            "and required action/object\nphrases. Each phrase should usually be 1-6 words. Do not include the whole "
            "question, leading question words, or broad\nanswer slots such as \"event\", \"date\", \"count\", "
            "\"place\", or \"duration\".\n"
        )
        target_support_schema = (
            '  "target_support_required": true,\n'
            '  "required_target_phrases": ["short exact target phrase from the question, or empty list"],\n'
        )
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {
            "role": "user",
            "content": ROUTER_TEMPLATE.format(
                question=example.question,
                question_date=example.question_date or "",
                target_support_instruction=target_support_instruction,
                target_support_schema=target_support_schema,
                temporal_hint_instruction=temporal_hint_instruction,
                temporal_hint_schema=temporal_hint_schema,
            ),
        },
    ]


def parse_route_classification(text: str) -> dict[str, Any]:
    parsed = extract_json_object(text) or {}
    task = str(parsed.get("task") or "single_fact").strip()
    operation = str(parsed.get("operation") or "none").strip()
    temporal_subtype = str(parsed.get("temporal_subtype") or "none").strip()
    answer_slot = str(parsed.get("answer_slot") or "fact").strip()
    if task not in TASKS:
        task = "single_fact"
    if operation not in OPERATIONS:
        operation = "none"
    if temporal_subtype not in TEMPORAL_SUBTYPES:
        temporal_subtype = "none"
    if answer_slot not in ANSWER_SLOTS:
        answer_slot = "fact"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    target_support_required = bool(parsed.get("target_support_required", False))
    raw_required_phrases = parsed.get("required_target_phrases") or []
    if not isinstance(raw_required_phrases, list):
        raw_required_phrases = []
    required_target_phrases = []
    for phrase in raw_required_phrases:
        phrase_text = " ".join(str(phrase).split())
        if phrase_text:
            required_target_phrases.append(phrase_text[:120])
        if len(required_target_phrases) >= 8:
            break
    raw_hints = parsed.get("temporal_search_hints") or []
    if not isinstance(raw_hints, list):
        raw_hints = []
    temporal_search_hints = []
    for hint in raw_hints:
        hint_text = " ".join(str(hint).split())
        if hint_text:
            temporal_search_hints.append(hint_text[:120])
        if len(temporal_search_hints) >= 6:
            break
    return {
        "task": task,
        "operation": operation,
        "temporal_subtype": temporal_subtype,
        "answer_slot": answer_slot,
        "target_support_required": target_support_required,
        "required_target_phrases": required_target_phrases,
        "temporal_search_hints": temporal_search_hints,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": str(parsed.get("rationale") or "")[:300],
    }

from __future__ import annotations

from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example, RetrievedChunk


EVIDENCE_RERANK_TEMPLATE = """Select the memory chunks that should be used to answer the user's question.

User Question:
{query}

Candidate Memories:
{context_str}

Selection Rules:
1. Keep chunks that directly support answering the question: facts, dates, quantities, events, user preferences, dislikes, constraints, goals, or owned resources.
2. For count, sum, duration, comparison, or ranking questions, keep every chunk that may provide an operand.
3. For recommendation or preference questions, keep user-stated preferences, anti-preferences, constraints, prior successful experiences, and relevant habits.
4. Prefer user statements over assistant suggestions. Assistant suggestions count only when the question asks about suggestions/plans or the user later accepted or confirmed them.
5. Exclude chunks that are topically similar but answer a different question, are duplicate evidence, or have the wrong scope/time.
6. Select at most {keep} chunks. Use fewer chunks if only a few are truly relevant.

Output Format:
{{
  "selected_ranks": [1, 4, 7],
  "reasoning": "Brief reason for the selected evidence"
}}

Return ONLY the JSON, no other text.
"""


REFLECTIVE_RETRIEVAL_TEMPLATE = """Generate focused retrieval queries from the first-pass evidence.

User Question:
{query}

First-pass Memories:
{context_str}

Requirements:
1. Generate queries that could retrieve missing or more precise evidence needed to answer the question.
2. Preserve exact names, dates, item types, units, actions, and time ranges from the question.
3. For count/list/sum questions, create queries for operands and distinct in-scope items.
4. For temporal questions, create queries for dated events, newest/previous states, and compared events.
5. For preference/recommendation questions, create queries for user preferences, dislikes, constraints, owned resources, and prior experiences.
6. Do not use or infer any reference answer. The queries must be based only on the user question and first-pass memories.
7. Return at most 3 short queries.

Output Format:
{{
  "queries": [
    "focused retrieval query"
  ],
  "keywords": [
    "important lexical keyword"
  ]
}}

Return ONLY the JSON, no other text.
"""


def evidence_rerank_messages(example: Example, retrieved: list[RetrievedChunk], keep: int) -> list[dict[str, str]]:
    prompt = EVIDENCE_RERANK_TEMPLATE.format(
        query=query_text(example),
        context_str=format_rerank_context(retrieved),
        keep=keep,
    )
    return [{"role": "user", "content": prompt}]


def reflective_retrieval_messages(example: Example, retrieved: list[RetrievedChunk]) -> list[dict[str, str]]:
    prompt = REFLECTIVE_RETRIEVAL_TEMPLATE.format(
        query=query_text(example),
        context_str=format_rerank_context(retrieved[:8]),
    )
    return [{"role": "user", "content": prompt}]


def parse_selected_ranks(raw_response: str) -> list[int]:
    value = extract_json_object(raw_response)
    if not value:
        return []
    ranks = value.get("selected_ranks")
    if not isinstance(ranks, list):
        return []
    selected = []
    for rank in ranks:
        try:
            selected.append(int(rank))
        except (TypeError, ValueError):
            continue
    return selected


def parse_reflective_queries(raw_response: str) -> tuple[list[str], list[str]]:
    value = extract_json_object(raw_response)
    if not value:
        return [], []
    return parse_string_list(value.get("queries"), 3), parse_string_list(value.get("keywords"), 12)


def parse_string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    parsed = []
    seen = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text.lower() in seen:
            continue
        parsed.append(text)
        seen.add(text.lower())
        if len(parsed) >= limit:
            break
    return parsed


def query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def format_rerank_context(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "None"
    blocks = []
    for chunk in retrieved:
        blocks.append(
            "\n".join(
                [
                    f"### Rank {chunk.rank}",
                    f"Date: {chunk.date}",
                    f"Session: {chunk.session_id}",
                    "Content:",
                    chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)

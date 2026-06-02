from __future__ import annotations

import hashlib

from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example, RetrievedChunk


ANSWER_TEMPLATE = """Answer the user's question based on the provided context.

User Question: {query}

Relevant Context: {context_str}

Requirements:
1. First, think through the reasoning process
2. Then provide a concise answer with only the essential personalized evidence
3. Answer must be based ONLY on the provided context
4. All dates in the response must be formatted as 'DD Month YYYY' but you can output more or less details if needed
5. Return your response in JSON format
{extra_requirements}

Output Format:
{{
  "reasoning": "One short sentence explaining the evidence used",
  "answer": "Concise answer with essential personalized evidence"
}}

Now answer the question. Return ONLY the JSON, no other text. Do not repeat yourself.
"""


GENERAL_REQUIREMENTS = """

Evidence-use requirements:
- First identify the specific evidence needed to answer the question.
- For recommendation or preference questions, use the user's stated preferences, dislikes, constraints, prior successful experiences, habits, and owned resources when they are relevant.
- Do not give generic suggestions when personalized evidence is available.
- For count, sum, difference, percentage, average, date duration, or comparison questions, verify operands and arithmetic before answering.
- Do not count the same event/item twice if it appears in multiple related memories.
- Do not treat missing evidence as zero unless the context explicitly supports that.
- Treat assistant suggestions as evidence only when the question asks about suggestions/plans or the user later accepted or confirmed them.
- If the context is insufficient, say so instead of inventing missing facts.
- In the final answer, be concise but include enough distinguishing detail to avoid a partial answer, such as full names, locations, dates, item names, or operands when the context provides them.
"""


PREFERENCE_REQUIREMENTS = """

Personalization requirements for recommendation or preference questions:
- Identify the user's relevant positive preferences, dislikes, constraints, owned tools/resources, prior successful experiences, and habits from the context.
- Do not give generic suggestions when personalized evidence is available.
- Apply known preferences to new situations when the connection is reasonable.
- Respect anti-preferences and avoid suggestions that conflict with them.
- If the exact requested item, place, or event is not in the context, answer with the type of option the user would likely prefer rather than inventing unsupported specific names.
- State the transferable preference or constraint explicitly, then give the recommendation only if it follows from that evidence.
- Do not introduce a named show, place, product, restaurant, or model unless that name appears in the context and directly matches the user's current request.
- In the final answer, include at most two key personalized reasons when they are needed to make the answer specific and verifiable.
"""


ASSISTANT_RECALL_REQUIREMENTS = """

Assistant-recall requirements:
- The user is asking about something the assistant previously said, suggested, recommended, called, wrote, or provided.
- Prefer evidence from assistant turns, especially the assistant reply paired with the user's earlier request.
- Answer by recalling the exact prior recommendation or wording as much as possible.
- Preserve alternatives, qualifiers, and option lists when they are part of the prior recommendation.
- Do not turn the answer into a new recommendation unless the question explicitly asks for new advice.
"""


EVIDENCE_TABLE_TEMPLATE = """Extract the evidence needed to answer the user's question.

User Question:
{query}

Retrieved Context:
{context_str}

Requirements:
1. Include every candidate evidence item from the context that may affect the answer.
2. Preserve exact numbers, dates, names, units, and event descriptions.
3. Set "include" to true only when the evidence explicitly satisfies the question's action, object, time range, and scope.
4. Do not treat owning, discussing, planning, liking, or asking about something as buying, acquiring, attending, completing, spending, or using it unless the evidence says so explicitly.
5. Use the same "canonical_item" for repeated mentions of the same fact. Count it once: include the clearest mention and mark duplicates as excluded.
6. Mark irrelevant, duplicate, out-of-scope, or underspecified items as excluded instead of deleting them.
7. If the question needs an operand that is missing from the context, set "sufficient" to false and explain the missing information.
8. Do not infer missing numbers from unrelated evidence.
9. For "current role" duration questions, distinguish total company tenure from time before promotion; if both are present, current-role duration is their difference.
10. For preference or recommendation questions, include user preferences, dislikes, constraints, habits, owned resources, and prior positive/negative experiences that transfer to the current question.
11. Include assistant-provided suggestions, estimates, recommendations, or hypothetical plans only if the user question asks about a plan or the user later confirms the event; otherwise exclude them.
12. For past user actions or experiences, assistant-estimated routes, prices, durations, or recommendations are not evidence that the user actually did or spent them.
13. For current/latest/recent/now questions, include older and newer candidates when available, then prefer the newest directly relevant evidence.
14. For previous/initial/original questions, answer the requested historical state instead of the newest state.
15. Preserve the scope and unit stated in the evidence; for example, do not convert "45 minutes each way" to "90 minutes" unless the question asks for a round-trip total.

Output Format:
{{
  "sufficient": true,
  "items": [
    {{
      "date": "DD Month YYYY or original date",
      "canonical_item": "Unique item/event/operand name",
      "evidence": "Short evidence text",
      "value": "Relevant number/name/date/unit if any",
      "include": true,
      "reason": "Why this item counts or does not count"
    }}
  ],
  "calculation": "Operands and arithmetic, or empty if no calculation is needed",
  "missing_info": "Required missing information, or empty if sufficient"
}}

Return ONLY the JSON, no other text.
"""


EVIDENCE_TABLE_ANSWER_TEMPLATE = """Answer the user's question using ONLY the extracted evidence JSON.

User Question:
{query}

Extracted Evidence JSON:
{evidence_json}

Requirements:
1. If "sufficient" is false because required information is missing, answer that the provided information is not enough.
2. Otherwise, use only included evidence items.
3. For count, sum, difference, percentage, average, date duration, or comparison questions, verify the operands and arithmetic.
4. Count each included "canonical_item" once, and do not count excluded or duplicate items.
5. For recommendation or preference questions, apply the included user preferences and constraints to the current question.
6. Provide a concise final answer with the correct unit and enough included evidence to make the answer complete.
7. For current/latest/recent/now questions, choose the newest directly relevant included evidence. For previous/initial/original questions, choose the requested older state.
8. For "how many" or count questions, answer with the count plus a short list of the counted items when item names are available.
9. For total/sum questions, answer with the total plus the operands used when values are available.

Output Format:
{{
  "reasoning": "One short sentence explaining the final calculation or decision",
  "answer": "Concise answer"
}}

Return ONLY the JSON, no other text. Do not repeat yourself.
"""


MULTI_EVIDENCE_TABLE_TEMPLATE = """Extract quote-grounded evidence for a count, list, sum, comparison, or aggregation question.

User Question:
{query}

Retrieved Context:
{context_str}

Rules:
1. Identify the exact target entity/action/time range/scope in the question before counting.
2. A memory counts only when it directly states an in-scope user fact or user-confirmed event.
3. Do not count assistant suggestions, hypothetical plans, examples, or merely discussed options unless the question asks about suggestions/plans.
4. Merge duplicate mentions of the same event/item/person.
5. Exclude related but out-of-scope evidence instead of deleting it.
6. Preserve exact quote, date, number, unit, and item name for every counted or excluded item.
7. If the question asks for a total/sum/difference/duration, list every operand and show the arithmetic.

Output Format:
{{
  "answer_type": "count|list|sum|difference|duration|comparison|other",
  "counted_items": [
    {{
      "canonical_item": "unique counted item/event/operand",
      "date": "date or empty",
      "quote": "short exact supporting quote",
      "value": "number/name/date/unit if relevant",
      "reason": "why it is in scope"
    }}
  ],
  "excluded_items": [
    {{
      "canonical_item": "excluded item/event",
      "date": "date or empty",
      "quote": "short quote",
      "reason": "duplicate|out_of_scope|assistant_suggestion|not_confirmed|wrong_time_range"
    }}
  ],
  "calculation": "explicit arithmetic or counting rule",
  "missing_info": "missing required evidence, or empty"
}}

Return ONLY the JSON, no other text.
"""


MULTI_EVIDENCE_ANSWER_TEMPLATE = """Answer the user's question using ONLY the extracted evidence JSON.

User Question:
{query}

Extracted Evidence JSON:
{evidence_json}

Rules:
1. Use counted_items and calculation only.
2. Do not include excluded_items in the final answer.
3. If no counted item is sufficient, say the provided information is not enough.
4. Return a concise answer with the number/list/value and one short evidence phrase if helpful.

Output Format:
{{
  "reasoning": "One short sentence explaining the evidence used",
  "answer": "Concise answer"
}}

Return ONLY the JSON, no other text.
"""


VERIFY_ANSWER_TEMPLATE = """Verify and finalize the draft answer using ONLY the retrieved context.

User Question:
{query}

Retrieved Context:
{context_str}

Draft Answer:
{draft_answer}

Rules:
1. If the draft answer is supported and complete, keep it.
2. If the draft answer picks the wrong item, date, order, quantity, or unit, correct it.
3. For order and date questions, verify the compared events and dates before finalizing.
4. If the context is insufficient, say that the provided information is not enough.

Output Format:
{{
  "reasoning": "One short sentence explaining the verification",
  "answer": "Final concise answer"
}}

Return ONLY the JSON, no other text.
"""


CATEGORY5_TEMPLATE = """Based on the context below, answer the following question.

Context:{context_str}

Question: {question}

Select the correct answer from the following two options. If the given answer is wrong or not answerable based on the context, you should choose "Not mentioned in the conversation".

Option A: {option_a}
Option B: {option_b}

Requirements:
1. Choose the option that best matches the context
2. If neither answer is supported by the context, or if the provided specific answer is incorrect, choose "Not mentioned in the conversation"
3. Return your response in JSON format

Output Format:
{{
  "reasoning": "Brief explanation of your choice",
  "answer": "Your selected answer"
}}

Return ONLY the JSON, no other text.
"""


def answer_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    requirement_style: str = "general",
    max_context_chars: int = 0,
) -> list[dict[str, str]]:
    context = format_context(retrieved, max_chars=max_context_chars)
    if is_locomo_category5(example):
        option_a, option_b = category5_options(example)
        prompt = CATEGORY5_TEMPLATE.format(
            context_str=context,
            question=example.question,
            option_a=option_a,
            option_b=option_b,
        )
    else:
        prompt = ANSWER_TEMPLATE.format(
            query=query_text(example),
            context_str=context,
            extra_requirements=answer_requirements(requirement_style),
        )
    return [{"role": "user", "content": prompt}]


def evidence_table_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    max_context_chars: int = 0,
) -> list[dict[str, str]]:
    prompt = EVIDENCE_TABLE_TEMPLATE.format(
        query=query_text(example),
        context_str=format_context(retrieved, max_chars=max_context_chars),
    )
    return [{"role": "user", "content": prompt}]


def evidence_table_answer_messages(example: Example, evidence_json: str) -> list[dict[str, str]]:
    prompt = EVIDENCE_TABLE_ANSWER_TEMPLATE.format(query=query_text(example), evidence_json=evidence_json.strip())
    return [{"role": "user", "content": prompt}]


def multi_evidence_table_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    max_context_chars: int = 0,
) -> list[dict[str, str]]:
    prompt = MULTI_EVIDENCE_TABLE_TEMPLATE.format(
        query=query_text(example),
        context_str=format_context(retrieved, max_chars=max_context_chars),
    )
    return [{"role": "user", "content": prompt}]


def multi_evidence_answer_messages(example: Example, evidence_json: str) -> list[dict[str, str]]:
    prompt = MULTI_EVIDENCE_ANSWER_TEMPLATE.format(query=query_text(example), evidence_json=evidence_json.strip())
    return [{"role": "user", "content": prompt}]


def verify_answer_messages(
    example: Example,
    retrieved: list[RetrievedChunk],
    draft_answer: str,
    *,
    max_context_chars: int = 0,
) -> list[dict[str, str]]:
    prompt = VERIFY_ANSWER_TEMPLATE.format(
        query=query_text(example),
        context_str=format_context(retrieved, max_chars=max_context_chars),
        draft_answer=draft_answer,
    )
    return [{"role": "user", "content": prompt}]


def query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def answer_requirements(style: str) -> str:
    if style == "assistant_recall":
        return ASSISTANT_RECALL_REQUIREMENTS
    if style == "preference":
        return PREFERENCE_REQUIREMENTS
    return GENERAL_REQUIREMENTS


def format_context(retrieved: list[RetrievedChunk], *, max_chars: int = 0) -> str:
    if not retrieved:
        return "None"
    blocks = []
    used = 0
    for chunk in retrieved:
        block = "\n".join(
            [
                f"### Memory {chunk.rank}",
                f"Date: {chunk.date}",
                "Content:",
                chunk.text,
            ]
        )
        if max_chars > 0 and blocks and used + len(block) > max_chars:
            break
        if max_chars > 0 and not blocks and len(block) > max_chars:
            block = block[:max_chars].rstrip() + "\n...[truncated]"
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def parse_answer(raw_response: str) -> str:
    value = extract_json_object(raw_response)
    if value and value.get("answer") is not None:
        return str(value["answer"]).strip()
    return raw_response.strip()


def is_locomo_category5(example: Example) -> bool:
    return example.dataset == "locomo" and str(example.metadata.get("category")) == "5"


def category5_options(example: Example) -> tuple[str, str]:
    not_mentioned = "Not mentioned in the conversation"
    adversarial = str(example.metadata.get("adversarial_answer") or "")
    options = [not_mentioned, adversarial or not_mentioned]
    digest = hashlib.md5(example.sample_id.encode("utf-8")).hexdigest()
    if int(digest, 16) % 2:
        options.reverse()
    return options[0], options[1]

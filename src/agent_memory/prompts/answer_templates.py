from __future__ import annotations


DIRECT_ANSWER_TEMPLATE = """Answer the user's question based on the provided context.

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
- If the context is insufficient, say so instead of inventing missing facts, and name the missing target constraint when it is clear.
- Match the requested answer slot exactly: where/place questions need a place, when/date questions need a time, who questions need a person, how-many questions need a count, and how-long questions need a duration. Do not answer with a related date, event, or explanation if the requested slot is different.
- For "what event", "what milestone", "what happened", "what did I do", or similar event-slot questions, answer the event/action itself. Include a date only as supporting detail, not as the main answer.
- In the final answer, be concise but include enough distinguishing detail to avoid a partial answer, such as full names, locations, dates, item names, or operands when the context provides them.
"""


ANSWER_DETAIL_REQUIREMENTS = """

Answer-detail requirements:
- Do not answer with only a bare number, name, or category when the evidence provides distinguishing details that are part of the answer.
- For count questions, include the count plus a compact parenthetical or short list of counted items when item names are available.
- For place, school, organization, or venue answers, preserve the full named place and available city, country, or institution qualifier.
- For indexed or list-recall questions, preserve the full recalled item text, not just its head noun or broad category.
- For time or date answers, include the concrete time/date and the event or state it applies to when that detail is available.
- Keep the answer concise; prefer one sentence unless the question asks for an ordered list.
"""


LIST_EVIDENCE_REQUIREMENTS = """
16. For list-style what/which questions, preserve every distinct in-scope item; do not collapse a list into a category unless the question asks for a category.
"""


LIST_EVIDENCE_ANSWER_REQUIREMENTS = """
10. For list-style what/which questions, answer with all distinct included item names or values, not just one example.
"""


PREFERENCE_REQUIREMENTS = """

Personalization requirements for recommendation or preference questions:
- Identify the user's relevant positive preferences, dislikes, constraints, owned tools/resources, prior successful experiences, and habits from the context.
- Prefer the most directly related prior turns on the same topic as the current request before using broader lifestyle preferences.
- When the user asks for advice, tips, or suggestions, anchor the answer in the user's named items, concrete past successes, stated goals, and follow-up interests from those directly related turns.
- Do not give generic suggestions when personalized evidence is available.
- Apply known preferences to new situations when the connection is reasonable.
- Respect anti-preferences and avoid suggestions that conflict with them.
- If the exact requested item, place, or event is not in the context, answer with the type of option the user would likely prefer rather than inventing unsupported specific names.
- State the transferable preference or constraint explicitly, then give the recommendation only if it follows from that evidence.
- Do not introduce a named show, place, product, restaurant, or model unless that name appears in the context and directly matches the user's current request.
- If the context supports only a preference profile and not a specific named option, answer with the preferred option type or selection criteria rather than naming a new hotel, brand, product, venue, show, restaurant, or model.
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


FACTUAL_EVIDENCE_EXTRACTION_TEMPLATE = """Extract the evidence needed to answer the user's question.

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
{aggregation_requirements}
{list_requirements}
{temporal_requirements}

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


FACTUAL_EVIDENCE_ANSWER_TEMPLATE = """Answer the user's question using ONLY the extracted evidence JSON.

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
10. Match the requested answer slot exactly: where/place questions need a place, when/date questions need a time, who questions need a person, how-many questions need a count, and how-long questions need a duration.
11. For "what event", "what milestone", "what happened", "what did I do", or similar event-slot questions, answer the event/action itself. Include a date only as supporting detail, not as the main answer.
{aggregation_requirements}
{list_requirements}
{temporal_requirements}

Output Format:
{{
  "reasoning": "One short sentence explaining the final calculation or decision",
  "answer": "Concise answer"
}}

Return ONLY the JSON, no other text. Do not repeat yourself.
"""


TEMPORAL_EXTRACTION_REQUIREMENTS = """

Temporal requirements:
- First identify the exact event/action/entity asked about, then select dates only for matching evidence.
- Treat the `Date:` field as the memory/session date. It is not always the event date.
- Prefer the event date or time phrase stated in `Content` over the memory/session `Date:` when they differ.
- If `Content` says yesterday, tomorrow, last Friday, last week, the week before, the weekend before, two weekends before, or a similar relative phrase, resolve it from the memory/session `Date:` and preserve the original phrase when useful.
- For planned future events, answer the planned time from the content, not the date when the plan was discussed.
- Include conflicting candidate dates in the evidence JSON and exclude the ones that do not match the asked event.
"""


DURATION_EXTRACTION_REQUIREMENTS = """

Duration requirements:
- For "how long", "for how long", "days/weeks/months/years", and "since" questions, first look for an explicit duration phrase in the content, such as "about an hour", "two hours", "four months", "six months", "nearly three months", or "a year ago".
- If an explicit duration phrase directly matches the target event or relationship, preserve that phrase and do not replace it with a date difference.
- Use date arithmetic only when the context gives both matching endpoints and no directly stated duration answers the question.
- Preserve the question's requested unit and approximate wording. Do not convert months or years into exact days unless the question asks for days.
- Do not use a memory/session date as an endpoint unless the content says the target event happened on that date.
"""


SCOPED_AGGREGATION_EXTRACTION_TEMPLATE = """Extract quote-grounded evidence for a count, list, sum, comparison, or aggregation question.

User Question:
{query}

Retrieved Context:
{context_str}

Rules:
1. Identify the exact target entity, owner/person, action, time range, and scope before counting.
2. A counted item must be directly stated by the in-scope user or explicitly confirmed as their fact/event.
3. Do not count assistant suggestions, congratulations, paraphrases, hypothetical plans, examples, or merely discussed options unless the question asks about suggestions/plans.
4. Merge repeated mentions of the same real-world item/event/process. Use one canonical_item for the same car, same child, same injury, same paperwork process, same trip, or same purchase even if it appears in multiple turns.
5. A later mention counts as a new item/event only if the quote explicitly distinguishes it as new, separate, another, second, different, or otherwise distinct from the earlier one.
6. For ownership counts, repeated references like "my car", "the Ferrari", "this beauty", or "my Ferrari" can be the same object. Count only distinct owned objects, not mentions.
7. For process counts, multiple turns about the same incident or workflow count once. For example, "dealing with insurance" and "the insurance process had paperwork" for the same accident are one insurance-paperwork instance.
8. For people/family counts, plural words such as "kids" establish at least two people, but a named relation such as "son" may be one of those kids unless the quote says he is additional.
9. For win/event counts, include only quotes that explicitly state a win/victory/trophy/buzzer-beater-to-win or another clear successful outcome. Exclude fan support, high scores, excitement, or a successful charity event unless the target win is explicit.
10. Exclude related but out-of-scope evidence instead of deleting it, and explain the exclusion.
11. Preserve exact quote, date, number, unit, and item name for every counted or excluded item.
12. If the evidence is ambiguous between a duplicate and a new item, mark it excluded as ambiguous_duplicate unless the question can be answered with a lower bound.
13. If the question asks for a total/sum/difference/duration, list every operand and show the arithmetic.

Output Format:
{{
  "answer_type": "count|list|sum|difference|duration|comparison|other",
  "target_scope": "owner/person + entity/action + time range/scope being counted",
  "counted_items": [
    {{
      "canonical_item": "unique counted item/event/operand",
      "date": "date or empty",
      "quote": "short exact supporting quote",
      "value": "number/name/date/unit if relevant",
      "reason": "why it is in scope and distinct from other counted items"
    }}
  ],
  "excluded_items": [
    {{
      "canonical_item": "excluded item/event",
      "date": "date or empty",
      "quote": "short quote",
      "reason": "duplicate|ambiguous_duplicate|out_of_scope|assistant_suggestion|not_confirmed|wrong_time_range|not_explicit"
    }}
  ],
  "calculation": "explicit arithmetic or counting rule",
  "missing_info": "missing required evidence, or empty"
}}

Return ONLY the JSON, no other text.
"""


SET_OPERATION_EXTRACTION_TEMPLATE = SCOPED_AGGREGATION_EXTRACTION_TEMPLATE.replace(
    "\nOutput Format:\n",
    """
14. First identify the requested set operation: count, list, sum, difference, duration, comparison, intersection, or union.
15. For intersection/commonality questions using words such as both, shared, similar, or common, list only items supported for every target subject. Evidence for just one subject is out of scope.
16. For intersection questions, use the shared item as canonical_item. The evidence field must include compact support for each target subject, for example "A: quote; B: quote".
17. For yes/no questions about whether two or more targets both satisfy a condition, include the condition only when every target has direct support. If one target is unsupported, make that explicit in excluded_items.
18. Do not replace a concrete shared item with a broad category when the context gives the concrete item name.

Output Format:
""",
)


SCOPED_AGGREGATION_ANSWER_TEMPLATE = """Answer the user's question using ONLY the extracted evidence JSON.

User Question:
{query}

Extracted Evidence JSON:
{evidence_json}

Rules:
1. Use counted_items and calculation only.
2. Count distinct real-world items/events/processes, not mentions. If counted_items repeat the same canonical_item, same quote, same date, or same underlying object/process, merge them before answering.
3. Do not include excluded_items in the final answer.
4. If the evidence is insufficient for an exact count, say the provided information is not enough unless the evidence supports a clear lower bound and the question can accept it.
5. For ownership counts, repeated references to the same object count once unless the evidence explicitly says another/new/second/different object.
6. For process counts, multiple turns about the same incident count once.
7. For people/family counts, avoid adding a relation to a plural group unless the evidence says the relation is additional.
8. Return a concise answer with the number/list/value and one short evidence phrase if helpful.
9. Match the requested answer slot exactly; do not answer with a related date, event, or explanation if the question asks for a place, person, count, duration, or concrete item.

Output Format:
{{
  "reasoning": "One short sentence explaining the distinct items/events counted",
  "answer": "Concise answer"
}}

Return ONLY the JSON, no other text.
"""


SET_OPERATION_ANSWER_TEMPLATE = SCOPED_AGGREGATION_ANSWER_TEMPLATE.replace(
    "\nOutput Format:\n",
    """
9. If the evidence JSON describes an intersection/commonality question, answer only with counted_items that are supported for every target subject.
10. Do not answer with one-sided facts from excluded_items. If no shared item is supported, say the provided information is not enough unless the evidence directly supports a yes/no "No".
11. Preserve concrete shared names, values, places, activities, or objects when they appear in the evidence.

Output Format:
""",
)


ANSWER_VERIFICATION_TEMPLATE = """Verify and finalize the draft answer using ONLY the retrieved context.

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

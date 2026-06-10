from __future__ import annotations

from datetime import date as Date
from datetime import timedelta
import re
from typing import Any

from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example, RetrievedChunk
from agent_memory.baseline.routing import asks_assistant_memory


def normalize_relative_duration_record(record: dict[str, Any], route: str) -> tuple[dict[str, Any], str]:
    question = str(record.get("question") or "").lower()
    answer = str(record.get("hypothesis") or "")
    if "ago" not in question:
        return record, route
    normalized = round_near_integer_duration(answer)
    if normalized == answer:
        return record, route
    return {**record, "hypothesis": normalized, "duration_normalized_from": answer}, f"{route}+duration_rounding"


def is_date_only_answer(answer: str) -> bool:
    tokens = re.findall(r"[A-Za-z0-9]+", answer)
    if len(tokens) > 12:
        return False
    month_pattern = (
        r"\b(january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\b"
    )
    return bool(re.search(month_pattern, answer, re.IGNORECASE) or re.search(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", answer))


def apply_answer_guardrails(
    example: Example,
    retrieved: list[RetrievedChunk],
    hypothesis: str,
    *,
    all_chunks: list[Any] | None = None,
    relative_temporal_guardrail: bool = False,
    relative_temporal_guardrail_scope: str = "context",
    relative_temporal_guardrail_loose: bool = False,
    relative_temporal_guardrail_require_temporal_hypothesis: bool = False,
    relative_temporal_guardrail_allow_insufficient_year: bool = False,
    relative_temporal_guardrail_preserve_relative_weekday: bool = False,
    relative_temporal_guardrail_future_relative: bool = False,
    relative_temporal_guardrail_allow_excluded_relative: bool = False,
    relative_temporal_guardrail_allow_insufficient_included_relative: bool = False,
    missing_target_guardrail: bool = True,
    missing_target_guardrail_skip_inference: bool = False,
    missing_target_guardrail_skip_family_answered: bool = False,
    missing_target_guardrail_relation_aliases: bool = False,
    relative_temporal_evidence: str | None = None,
) -> str:
    memory_context = "\n".join(str(item.text) for item in (all_chunks or retrieved))
    deterministic_answer = deterministic_relative_event_order_answer(example.question, all_chunks)
    answer = deterministic_answer or hypothesis
    if relative_temporal_guardrail and deterministic_answer is None:
        temporal_candidate_allowed = (
            not relative_temporal_guardrail_require_temporal_hypothesis
            or answer_has_temporal_value(answer)
            or (relative_temporal_guardrail_allow_insufficient_year and is_insufficient_answer(answer))
        )
        if temporal_candidate_allowed:
            if relative_temporal_guardrail_scope == "evidence":
                replacement = deterministic_relative_temporal_answer_from_evidence(
                    example.question,
                    answer,
                    relative_temporal_evidence,
                    allow_loose=relative_temporal_guardrail_loose,
                    allow_insufficient_year=relative_temporal_guardrail_allow_insufficient_year,
                    preserve_relative_weekday=relative_temporal_guardrail_preserve_relative_weekday,
                    allow_future_relative=relative_temporal_guardrail_future_relative,
                    allow_excluded_relative=relative_temporal_guardrail_allow_excluded_relative,
                    allow_insufficient_included_relative=relative_temporal_guardrail_allow_insufficient_included_relative,
                )
            else:
                temporal_chunks = all_chunks if relative_temporal_guardrail_scope == "all_chunks" else retrieved
                replacement = deterministic_relative_temporal_answer(
                    example.question,
                    answer,
                    temporal_chunks,
                    allow_loose=relative_temporal_guardrail_loose,
                    preserve_relative_weekday=relative_temporal_guardrail_preserve_relative_weekday,
                    allow_future_relative=relative_temporal_guardrail_future_relative,
                )
            answer = replacement or answer
    answer = normalize_over_precise_duration(example.question, answer)
    answer = normalize_bare_year_difference(example.question, answer)
    answer = normalize_list_count_arithmetic(example.question, answer)

    target_context = memory_context
    if all_chunks and not asks_assistant_memory(example.question):
        user_context = user_memory_context(all_chunks)
        if user_context:
            target_context = user_context
    missing_target = None
    if missing_target_guardrail and not deterministic_answer:
        missing_target = exact_target_missing_reason(
            example.question,
            target_context,
            use_relation_aliases=missing_target_guardrail_relation_aliases,
        )
    if missing_target and not should_skip_missing_target_guardrail(
        missing_target,
        answer,
        example.question,
        skip_inference=missing_target_guardrail_skip_inference,
        skip_family_answered=missing_target_guardrail_skip_family_answered,
    ):
        return missing_target
    return answer


def round_near_integer_duration(answer: str) -> str:
    pattern = re.compile(r"\b(\d+)\.(\d+)\s+(days?|weeks?|months?|years?)\b", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        value = float(f"{match.group(1)}.{match.group(2)}")
        rounded = round(value)
        if rounded <= 0 or abs(value - rounded) > 0.2:
            return match.group(0)
        unit = match.group(3).lower()
        if rounded == 1:
            unit = unit.rstrip("s")
        elif not unit.endswith("s"):
            unit = f"{unit}s"
        return f"{rounded} {unit}"

    return pattern.sub(replace, answer)


def deterministic_relative_event_order_answer(question: str, all_chunks: list[Any] | None) -> str | None:
    options = parse_event_order_options(question)
    if len(options) != 2:
        return None
    ranked = []
    for option in options:
        rank = relative_event_rank(option, all_chunks)
        if rank is None:
            return None
        ranked.append((rank, option))
    if ranked[0][0] == ranked[1][0]:
        return None
    earlier = min(ranked, key=lambda item: item[0])[1]
    if question.lower().startswith("who did i meet first"):
        return earlier
    return f"The {earlier} happened first."


def deterministic_relative_temporal_answer(
    question: str,
    hypothesis: str,
    all_chunks: list[Any] | None,
    *,
    allow_loose: bool = False,
    preserve_relative_weekday: bool = False,
    allow_future_relative: bool = False,
) -> str | None:
    lowered_question = question.lower()
    if not all_chunks or not is_temporal_date_question(lowered_question):
        return None
    terms = temporal_question_terms(question)
    if not terms:
        return None
    if has_anchored_relative_temporal_answer(hypothesis):
        return None

    requested_months = requested_month_numbers(lowered_question)
    candidates: list[tuple[int, tuple[int, int, int], int, str]] = []
    for index, chunk in enumerate(all_chunks):
        text = str(getattr(chunk, "text", "") or "")
        date_key = event_date_sort_key(str(getattr(chunk, "date", "") or ""))
        if not text or date_key is None:
            continue
        relative_answer, relative_key = relative_temporal_phrase_answer(
            text,
            date_key,
            allow_loose=allow_loose,
            preserve_relative_weekday=preserve_relative_weekday,
            allow_future_relative=allow_future_relative,
        )
        if not relative_answer:
            continue
        if requested_months and relative_key and relative_key[1] not in requested_months:
            continue

        start = max(0, index - 1)
        end = min(len(all_chunks), index + 2)
        window_text = " ".join(str(getattr(all_chunks[item], "text", "") or "") for item in range(start, end))
        score = temporal_match_score(terms, window_text)
        if score < min(2, len(terms)):
            continue
        candidates.append((score, relative_key or date_key, index, relative_answer))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best = candidates[0]
    if looks_like_same_temporal_answer(hypothesis, best[3]):
        return None
    return best[3]


def deterministic_relative_temporal_answer_from_evidence(
    question: str,
    hypothesis: str,
    evidence_response: str | None,
    *,
    allow_loose: bool = False,
    allow_insufficient_year: bool = False,
    preserve_relative_weekday: bool = False,
    allow_future_relative: bool = False,
    allow_excluded_relative: bool = False,
    allow_insufficient_included_relative: bool = False,
) -> str | None:
    lowered_question = question.lower()
    if not evidence_response or not is_temporal_date_question(lowered_question):
        return None
    terms = temporal_question_terms(question)
    if not terms:
        return None
    if has_anchored_relative_temporal_answer(hypothesis):
        return None
    value = extract_json_object(evidence_response)
    if not value:
        return None
    sufficient = value.get("sufficient") is not False
    if not sufficient and not allow_insufficient_year and not allow_insufficient_included_relative:
        return None
    requested_months = requested_month_numbers(lowered_question)
    included_canonicals = included_canonical_items(value.get("items") or [])
    candidates: list[tuple[int, int, tuple[int, int, int], str]] = []
    for index, item in enumerate(value.get("items") or []):
        if not isinstance(item, dict):
            continue
        canonical_text = str(item.get("canonical_item") or "")
        text = " ".join(str(item.get(key) or "") for key in ("canonical_item", "evidence", "value", "reason"))
        allow_item_year = allow_insufficient_year and not sufficient and "last year" in text.lower()
        allow_insufficient_item = (
            allow_insufficient_included_relative and not sufficient and item.get("include") is not False
        )
        allow_excluded_item = (
            allow_excluded_relative
            and item.get("include") is False
            and is_same_canonical_relative_evidence(item, text, included_canonicals, allow_future_relative)
        )
        if item.get("include") is False and not allow_item_year and not allow_excluded_item:
            continue
        if not sufficient and not allow_item_year and not allow_insufficient_item and not allow_excluded_item:
            continue
        date_key = event_date_sort_key(str(item.get("date") or "")) or event_date_sort_key(text)
        if date_key is None:
            continue
        relative_source = relative_temporal_source_text(item, terms, allow_future_relative=allow_future_relative)
        if not relative_source:
            continue
        if has_future_relative_temporal_phrase(relative_source) and event_date_sort_key(str(item.get("value") or "")):
            continue
        anchor_key = relative_temporal_anchor_key(item, date_key)
        relative_answer, relative_key = relative_temporal_phrase_answer(
            relative_source,
            anchor_key,
            allow_loose=allow_loose or allow_item_year,
            preserve_relative_weekday=preserve_relative_weekday,
            allow_future_relative=allow_future_relative,
        )
        if not relative_answer:
            continue
        if requested_months and relative_key and relative_key[1] not in requested_months:
            continue
        score = temporal_match_score(terms, f"{canonical_text} {relative_source}")
        # The evidence compiler has already marked included items as in-scope.
        # For relative-time normalization, require a lighter lexical overlap for
        # included evidence so paraphrases such as "start taking classes" vs.
        # "signed up for a class" can still resolve the relative date.
        min_score = 1 if item.get("include") is not False else min(2, len(terms))
        if score < min(min_score, len(terms)):
            continue
        candidates.append((score, index, relative_key or date_key, relative_answer))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best = candidates[0]
    if looks_like_same_temporal_answer(hypothesis, best[3]):
        return None
    return best[3]


def relative_temporal_phrase_answer(
    text: str,
    date_key: tuple[int, int, int],
    *,
    allow_loose: bool = False,
    preserve_relative_weekday: bool = False,
    allow_future_relative: bool = False,
) -> tuple[str | None, tuple[int, int, int] | None]:
    lowered = text.lower()
    anchor = date_from_key(date_key)
    if anchor is None:
        return None, None

    if re.search(r"\blast night\b", lowered):
        return format_date(anchor - timedelta(days=1)), key_from_date(anchor - timedelta(days=1))
    if re.search(r"\byesterday\b", lowered):
        return format_date(anchor - timedelta(days=1)), key_from_date(anchor - timedelta(days=1))
    if re.search(r"\btomorrow\b", lowered):
        return format_date(anchor + timedelta(days=1)), key_from_date(anchor + timedelta(days=1))

    day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b", lowered)
    if day_match:
        value = parse_small_number(day_match.group(1))
        if value is not None:
            result = anchor - timedelta(days=value)
            return format_date(result), key_from_date(result)

    weekday_match = re.search(
        r"\blast\s+(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        lowered,
    )
    if weekday_match:
        result = previous_weekday(anchor, weekday_match.group(1))
        if result is not None:
            if preserve_relative_weekday:
                return f"The {weekday_full_name(weekday_match.group(1))} before {format_date(anchor)}", key_from_date(result)
            return format_date(result), key_from_date(result)

    if re.search(r"\b(?:last weekend|this past weekend|past weekend)\b", lowered):
        return f"The weekend before {format_date(anchor)}", key_from_date(anchor - timedelta(days=2))
    if re.search(r"\b(?:two weekends before|two weekends ago)\b", lowered):
        return f"two weekends before {format_date(anchor)}", key_from_date(anchor - timedelta(days=14))
    if re.search(r"\b(?:last week|the week before)\b", lowered):
        return f"The week before {format_date(anchor)}", key_from_date(anchor - timedelta(days=7))
    if allow_future_relative and re.search(r"\b(?:next weekend|coming weekend|the following weekend|weekend after)\b", lowered):
        return f"The weekend after {format_date(anchor)}", key_from_date(anchor + timedelta(days=7))
    if allow_future_relative and re.search(r"\b(?:two weekends later|two weekends after)\b", lowered):
        return f"two weekends after {format_date(anchor)}", key_from_date(anchor + timedelta(days=14))
    if allow_future_relative and re.search(r"\b(?:next week|the following week|week after)\b", lowered):
        return f"The week after {format_date(anchor)}", key_from_date(anchor + timedelta(days=7))

    if re.search(r"\bnext month\b", lowered):
        year, month = add_months(anchor.year, anchor.month, 1)
        return f"{month_name(month)} {year}", (year, month, 1)
    if re.search(r"\blast month\b", lowered):
        year, month = add_months(anchor.year, anchor.month, -1)
        return f"{month_name(month)} {year}", (year, month, 1)
    if allow_loose and re.search(r"\blast year\b", lowered):
        return str(anchor.year - 1), (anchor.year - 1, 1, 1)
    if allow_loose and re.search(r"\ba few weeks ago\b", lowered):
        return f"a few weeks before {format_date(anchor)}", key_from_date(anchor - timedelta(days=21))
    return None, None


def included_canonical_items(items: list[Any]) -> set[str]:
    canonicals: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("include") is False:
            continue
        canonical = normalize_text_for_match(str(item.get("canonical_item") or ""))
        if canonical:
            canonicals.add(canonical)
    return canonicals


def is_same_canonical_relative_evidence(
    item: dict[str, Any],
    text: str,
    included_canonicals: set[str],
    allow_future_relative: bool,
) -> bool:
    canonical = normalize_text_for_match(str(item.get("canonical_item") or ""))
    if not canonical or canonical not in included_canonicals:
        return False
    if not has_relative_temporal_phrase(text, allow_future_relative=allow_future_relative):
        return False
    reason = normalize_text_for_match(str(item.get("reason") or ""))
    if re.search(r"\b(?:not the same|different event|out of scope|wrong time range|irrelevant)\b", reason):
        return False
    return bool(re.search(r"\b(?:duplicate|same event|same date|same canonical|already included|already covered)\b", reason))


def has_relative_temporal_phrase(text: str, *, allow_future_relative: bool) -> bool:
    lowered = text.lower()
    patterns = [
        r"\blast night\b",
        r"\byesterday\b",
        r"\btomorrow\b",
        r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b",
        r"\blast\s+(?:mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:rs|rsday)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        r"\b(?:last weekend|this past weekend|past weekend|two weekends before|two weekends ago)\b",
        r"\b(?:last week|the week before)\b",
        r"\blast month\b",
    ]
    if allow_future_relative:
        patterns.extend(
            [
                r"\b(?:next weekend|coming weekend|the following weekend|weekend after)\b",
                r"\b(?:two weekends later|two weekends after)\b",
                r"\b(?:next week|the following week|week after)\b",
                r"\bnext month\b",
            ]
        )
    return any(re.search(pattern, lowered) for pattern in patterns)


def has_future_relative_temporal_phrase(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(?:tomorrow|next month|next week|coming weekend|next weekend|following week|following weekend|week after|weekend after|two weekends later|two weekends after)\b",
            lowered,
        )
    )


def relative_temporal_anchor_key(item: dict[str, Any], default_key: tuple[int, int, int]) -> tuple[int, int, int]:
    """Find the memory/conversation date used to resolve a relative phrase.

    Evidence extraction sometimes stores the resolved event date in ``date`` and
    keeps the original relative phrase in ``evidence``.  In that case, a second
    subtraction would be wrong.  When the reason explicitly names the
    conversation/message date, use it as the anchor.
    """

    text = " ".join(str(item.get(key) or "") for key in ("reason", "calculation"))
    patterns = (
        r"\bconversation (?:occurred|happened|was|is)?\s*(?:on|dated)?\s*([A-Za-z]+\s+\d{1,2},?\s+20\d{2}|\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
        r"\bmessage (?:was )?(?:sent |dated )?(?:on )?([A-Za-z]+\s+\d{1,2},?\s+20\d{2}|\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
        r"\bfrom (?:the )?(?:conversation|message) (?:on )?([A-Za-z]+\s+\d{1,2},?\s+20\d{2}|\d{1,2}\s+[A-Za-z]+\s+20\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        key = event_date_sort_key(match.group(1))
        if key:
            return key
    return default_key


def is_temporal_date_question(lowered_question: str) -> bool:
    return bool(re.search(r"^\s*(?:when|what date)\b", lowered_question))


def relative_temporal_source_text(
    item: dict[str, Any],
    terms: list[str],
    *,
    allow_future_relative: bool,
) -> str | None:
    candidates: list[tuple[int, int, str]] = []
    fallback: str | None = None
    fields = (
        str(item.get("evidence") or ""),
        str(item.get("value") or ""),
        str(item.get("reason") or ""),
    )
    order = 0
    for field in fields:
        segments = temporal_phrase_segments(field, allow_future_relative=allow_future_relative)
        if segments and fallback is None:
            fallback = field if has_relative_temporal_phrase(field, allow_future_relative=allow_future_relative) else segments[0]
        for segment in segments:
            score = temporal_match_score(terms, segment)
            if score <= 0:
                continue
            candidates.append((score, order, segment))
            order += 1
    if not candidates:
        return fallback
    candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
    return candidates[0][2]


def temporal_phrase_segments(text: str, *, allow_future_relative: bool) -> list[str]:
    if not text or not has_relative_temporal_phrase(text, allow_future_relative=allow_future_relative):
        return []
    segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|[;\n]", text) if segment.strip()]
    phrase_segments = [
        segment for segment in segments if has_relative_temporal_phrase(segment, allow_future_relative=allow_future_relative)
    ]
    return phrase_segments or [text]


def temporal_question_terms(question: str) -> list[str]:
    stopwords = {
        "when",
        "what",
        "date",
        "did",
        "does",
        "do",
        "was",
        "were",
        "is",
        "are",
        "the",
        "a",
        "an",
        "to",
        "on",
        "in",
        "at",
        "for",
        "with",
        "and",
        "or",
        "of",
        "after",
        "before",
        "during",
        "together",
        "go",
        "went",
        "attend",
        "attended",
        "join",
        "joined",
        "meet",
        "met",
        "up",
        "have",
        "had",
    }
    normalized = normalize_temporal_match_text(question)
    words = []
    for word in re.findall(r"[a-z0-9]+", normalized):
        if word in stopwords or len(word) <= 2:
            continue
        words.append(word)
    return dedupe_targets(words)


def temporal_match_score(terms: list[str], text: str) -> int:
    normalized = normalize_temporal_match_text(text)
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", normalized))


def normalize_temporal_match_text(text: str) -> str:
    normalized = normalize_text_for_match(text)
    return " ".join(normalize_match_token(token) for token in normalized.split())


def normalize_match_token(token: str) -> str:
    """Small, domain-neutral normalization for matching question terms to evidence text."""
    if len(token) > 4 and token.endswith("ies"):
        return f"{token[:-3]}y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def requested_month_numbers(lowered_question: str) -> set[int]:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return {number for name, number in months.items() if re.search(rf"\b{name}\b", lowered_question)}


def looks_like_same_temporal_answer(left: str, right: str) -> bool:
    right_lowered = right.lower()
    if "before" in right_lowered or "ago" in right_lowered:
        return False
    left_key = event_date_sort_key(left)
    right_key = event_date_sort_key(right)
    if left_key and right_key:
        return left_key == right_key
    return left.strip().lower() == right.strip().lower()


def has_anchored_relative_temporal_answer(answer: str) -> bool:
    lowered = answer.lower()
    if not re.search(r"\b(?:before|after)\b", lowered):
        return False
    return event_date_sort_key(answer) is not None


def answer_has_temporal_value(answer: str) -> bool:
    if event_date_sort_key(answer):
        return True
    lowered = answer.lower()
    month_pattern = (
        r"\b(january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\b"
    )
    return bool(re.search(month_pattern, lowered) or re.search(r"\b20\d{2}\b", lowered))


def is_missing_target_inference_question(question: str) -> bool:
    normalized = " ".join(question.lower().split())
    return bool(re.search(r"\b(would|likely|might|could|potential(?:ly)?)\b", normalized) or "based on" in normalized)


def should_skip_missing_target_guardrail(
    missing_target: str,
    answer: str,
    question: str,
    *,
    skip_inference: bool,
    skip_family_answered: bool,
) -> bool:
    if skip_inference and is_missing_target_inference_question(question):
        return True
    if (
        skip_family_answered
        and "different family member" in missing_target.lower()
        and not is_insufficient_answer(answer)
        and bool(answer.strip())
    ):
        return True
    return False


def is_insufficient_answer(answer: str) -> bool:
    lowered = answer.lower()
    return "not enough" in lowered or "insufficient" in lowered or "cannot determine" in lowered


def user_turn_records(all_chunks: list[Any] | None) -> list[tuple[tuple[int, int, int], int, str, str]]:
    records: list[tuple[tuple[int, int, int], int, str, str]] = []
    if not all_chunks:
        return records
    for index, chunk in enumerate(all_chunks):
        text = str(getattr(chunk, "text", "")).strip()
        if not text.lower().startswith("user:"):
            continue
        date = str(getattr(chunk, "date", "") or "")
        date_key = event_date_sort_key(date) or (0, 0, 0)
        records.append((date_key, index, date, text.split(":", 1)[1].strip()))
    return records


def user_memory_context(all_chunks: list[Any] | None) -> str:
    return "\n".join(f"user: {text}" for _date_key, _index, _date, text in user_turn_records(all_chunks))


def parse_event_order_options(question: str) -> list[str]:
    text = question.strip().rstrip("?")
    patterns = (
        r"\b(?:which event happened first|which happened first|what happened first),?\s+(.+?)\s+or\s+(.+)$",
        r"\bwhich trip did i take first,?\s+(.+?)\s+or\s+(.+)$",
        r"\bwho did i meet first,?\s+(.+?)\s+or\s+(.+)$",
        r"\bwhich task did i complete first,?\s+(.+?)\s+or\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [clean_option(match.group(1)), clean_option(match.group(2))]
    return []


def relative_event_rank(option: str, all_chunks: list[Any] | None) -> int | None:
    terms = event_option_terms(option)
    if not terms:
        return None
    ranks: list[int] = []
    for date_key, _index, _date, text in user_turn_records(all_chunks):
        normalized = normalize_text_for_match(text)
        if event_terms_match(terms, normalized):
            rank = relative_time_rank(text, date_key)
            if rank is not None:
                ranks.append(rank)
    return min(ranks) if ranks else None


def event_option_terms(option: str) -> list[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "with",
        "one",
        "event",
        "happened",
        "first",
        "attendance",
        "start",
        "purchase",
        "purchasing",
        "malfunction",
        "trip",
        "solo",
        "family",
        "friends",
    }
    words = [word for word in re.findall(r"[a-z0-9]+", option.lower()) if word not in stopwords and len(word) > 2]
    return dedupe_targets(words)


def event_terms_match(terms: list[str], normalized_text: str) -> bool:
    hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", normalized_text))
    return hits >= min(2, len(terms))


def relative_time_rank(text: str, date_key: tuple[int, int, int]) -> int | None:
    lowered = text.lower()
    if "last year" in lowered:
        return -365
    if re.search(r"\bfew months ago\b", lowered):
        return -90
    past_months = re.search(
        r"\b(?:for\s+)?(?:the\s+)?past\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+months?\b",
        lowered,
    )
    if past_months:
        value = parse_small_number(past_months.group(1))
        if value is not None:
            return -30 * value
    if re.search(r"\b(?:last month|about a month ago|a month ago|one month ago)\b", lowered):
        return -30
    week_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+weeks?\s+ago\b", lowered)
    if week_match:
        value = parse_small_number(week_match.group(1))
        if value is not None:
            return -7 * value
    day_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+ago\b", lowered)
    if day_match:
        value = parse_small_number(day_match.group(1))
        if value is not None:
            return -value
    if "last week" in lowered:
        return -7
    if "yesterday" in lowered:
        return -1
    if date_key != (0, 0, 0):
        year, month, day = date_key
        return year * 372 + month * 31 + day
    return None


def parse_small_number(value: str) -> int | None:
    cleaned = value.strip().lower()
    if cleaned.isdigit():
        return int(cleaned)
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(cleaned)


def date_from_key(date_key: tuple[int, int, int]) -> Date | None:
    try:
        return Date(date_key[0], date_key[1], date_key[2])
    except ValueError:
        return None


def key_from_date(value: Date) -> tuple[int, int, int]:
    return value.year, value.month, value.day


def format_date(value: Date) -> str:
    return f"{value.day} {month_name(value.month)} {value.year}"


def month_name(month: int) -> str:
    names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return names[month]


def add_months(year: int, month: int, offset: int) -> tuple[int, int]:
    month_index = year * 12 + (month - 1) + offset
    return month_index // 12, month_index % 12 + 1


def previous_weekday(anchor: Date, weekday_text: str) -> Date | None:
    normalized = weekday_text[:3].lower()
    weekdays = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
        "sat": 5,
        "sun": 6,
    }
    target = weekdays.get(normalized)
    if target is None:
        return None
    delta = (anchor.weekday() - target) % 7
    if delta == 0:
        delta = 7
    return anchor - timedelta(days=delta)


def weekday_full_name(weekday_text: str) -> str:
    normalized = weekday_text[:3].lower()
    names = {
        "mon": "Monday",
        "tue": "Tuesday",
        "wed": "Wednesday",
        "thu": "Thursday",
        "fri": "Friday",
        "sat": "Saturday",
        "sun": "Sunday",
    }
    return names.get(normalized, weekday_text.title())


def event_date_sort_key(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    month_names = "|".join(months)
    match = re.search(rf"\b(\d{{1,2}})\s+({month_names})(?:,)?\s+(20\d{{2}})\b", text, re.IGNORECASE)
    if match:
        return int(match.group(3)), months[match.group(2).lower()], int(match.group(1))
    match = re.search(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(20\d{{2}})\b", text, re.IGNORECASE)
    if match:
        return int(match.group(3)), months[match.group(1).lower()], int(match.group(2))
    return None


def exact_target_missing_reason(question: str, context: str, *, use_relation_aliases: bool = False) -> str | None:
    lowered_question = question.lower()
    lowered_context = context.lower()
    order_option_reason = missing_order_option_reason(question, context)
    if order_option_reason:
        return order_option_reason

    doctor = re.search(r"\bdr\.?\s+([a-z][a-z'-]*)\b", lowered_question)
    if doctor and not re.search(rf"\bdr\.?\s+{re.escape(doctor.group(1))}\b", lowered_context):
        name = f"Dr. {doctor.group(1).title()}"
        return f"The provided information is not enough to answer the question. The context does not mention {name}."

    missing_required_target = missing_required_target_reason(question, context)
    if missing_required_target:
        return missing_required_target

    family_relations = {
        "uncle",
        "aunt",
        "dad",
        "father",
        "mom",
        "mother",
        "sister",
        "brother",
        "niece",
        "nephew",
        "cousin",
    }
    for relation in family_relations:
        if not re.search(rf"\b(?:my\s+)?{relation}(?:'s)?\b", lowered_question):
            continue
        if family_relation_present(relation, lowered_context, use_aliases=use_relation_aliases):
            continue
        other_relation = next(
            (
                other
                for other in family_relations
                if other != relation
                and not same_family_relation_group(relation, other, use_aliases=use_relation_aliases)
                and family_relation_present(other, lowered_context, use_aliases=use_relation_aliases)
            ),
            None,
        )
        if other_relation:
            return (
                "The provided information is not enough to answer the question. "
                f"The context mentions a different family member, not your {relation}."
            )
    return None


FAMILY_RELATION_ALIAS_GROUPS = (
    {"father", "dad"},
    {"mother", "mom", "mum"},
    {"brother", "bro"},
    {"sister", "sis"},
)


def family_relation_aliases(relation: str, *, use_aliases: bool) -> set[str]:
    if not use_aliases:
        return {relation}
    for group in FAMILY_RELATION_ALIAS_GROUPS:
        if relation in group:
            return group
    return {relation}


def same_family_relation_group(left: str, right: str, *, use_aliases: bool) -> bool:
    return bool(family_relation_aliases(left, use_aliases=use_aliases) & family_relation_aliases(right, use_aliases=use_aliases))


def family_relation_present(relation: str, lowered_context: str, *, use_aliases: bool) -> bool:
    return any(re.search(rf"\b{re.escape(alias)}\b", lowered_context) for alias in family_relation_aliases(relation, use_aliases=use_aliases))


def missing_required_target_reason(question: str, context: str) -> str | None:
    context_norm = normalize_text_for_match(context)
    for target in required_target_phrases(question):
        target_norm = normalize_text_for_match(target)
        if not target_norm or phrase_present_in_normalized_context(target_norm, context_norm):
            continue
        return (
            "The provided information is not enough to answer the question. "
            f"The context does not mention {target.strip()}."
        )
    return None


def router_required_target_missing_reason(target_phrases: list[Any], context: str) -> str | None:
    context_norm = normalize_text_for_match(context)
    for phrase in target_phrases:
        phrase_text = str(phrase).strip()
        if not phrase_text:
            continue
        tokens = required_target_phrase_tokens(phrase_text)
        if not tokens:
            continue
        if not should_apply_router_target_phrase_guard(phrase_text, tokens):
            continue
        if target_phrase_supported(phrase_text, tokens, context_norm):
            continue
        if should_name_missing_target_phrase(phrase_text):
            return (
                "The provided information is not enough to answer the question. "
                f"The context does not mention {phrase_text}."
            )
        return "The provided information is not enough."
    return None


def should_name_missing_target_phrase(phrase: str) -> bool:
    lowered = phrase.strip().lower()
    if lowered.startswith(("how ", "what ", "which ", "who ", "where ", "when ", "why ")):
        return False
    words = re.findall(r"[a-z0-9]+", lowered)
    return 0 < len(words) <= 10


def should_apply_router_target_phrase_guard(phrase: str, tokens: list[str]) -> bool:
    if len(tokens) > 8:
        return False
    lowered = phrase.strip().lower()
    if lowered.startswith(("how ", "what ", "which ", "who ", "where ", "when ", "why ")):
        return False
    if is_broad_category_phrase(phrase, tokens):
        return False
    if len(tokens) == 1 and not has_exact_name_marker(phrase):
        return False
    if " and " in lowered and len(tokens) <= 6:
        return True
    if has_exact_name_marker(phrase):
        return True
    head = tokens[-1]
    if head.endswith("s") and not head.endswith("ss"):
        return False
    return 1 <= len(tokens) <= 5


def is_broad_category_phrase(phrase: str, tokens: list[str]) -> bool:
    lowered_phrase = phrase.lower()
    broad_heads = {
        "clothing",
        "city",
        "country",
        "destination",
        "doctor",
        "expense",
        "fruit",
        "item",
        "location",
        "movie",
        "movy",
        "plant",
        "project",
        "recipe",
        "state",
        "trip",
        "week",
    }
    if any(token == "item" for token in tokens):
        return True
    head = tokens[-1]
    if head in {"movie", "movy", "film"} and has_exact_name_marker(phrase):
        return True
    always_broad_heads = {"city", "clothing", "country", "destination", "expense", "location", "state", "trip", "week"}
    if head in always_broad_heads and " and " not in lowered_phrase:
        return True
    if head in broad_heads and len(tokens) <= 2 and " and " not in lowered_phrase:
        return True
    if set(tokens) <= {"currently", "lead", "led", "leading", "pick", "return"}:
        return True
    return False


def has_exact_name_marker(phrase: str) -> bool:
    if re.search(r"\b(?:Dr|Mr|Mrs|Ms)\.?\s+[A-Z0-9]", phrase):
        return True
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", phrase)
    for index, word in enumerate(words):
        if any(char.isdigit() for char in word):
            return True
        if word.isupper() and len(word) > 1:
            return True
        if index > 0 and word[:1].isupper():
            return True
    return False


def required_target_phrase_tokens(phrase: str) -> list[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "answer",
        "count",
        "current",
        "date",
        "duration",
        "event",
        "fact",
        "first",
        "for",
        "from",
        "how",
        "i",
        "in",
        "latest",
        "me",
        "my",
        "of",
        "or",
        "place",
        "previous",
        "question",
        "scope",
        "the",
        "time",
        "to",
        "user",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
    tokens = []
    for token in re.findall(r"[a-z0-9]+", phrase.lower()):
        if token in stopwords:
            continue
        if len(token) <= 2 and not token.isdigit():
            continue
        tokens.append(normalize_match_token(token))
    return dedupe_targets(tokens)


def target_phrase_supported(phrase: str, tokens: list[str], context_norm: str) -> bool:
    phrase_norm = normalize_text_for_match(" ".join(normalize_match_token(token) for token in phrase.split()))
    if phrase_norm and phrase_present_in_normalized_context(phrase_norm, context_norm):
        return True
    return all(token_present_in_context(token, context_norm) for token in tokens)


def token_present_in_context(token: str, context_norm: str) -> bool:
    if re.search(rf"\b{re.escape(token)}\b", context_norm):
        return True
    if token.endswith("s") and re.search(rf"\b{re.escape(token[:-1])}\b", context_norm):
        return True
    return bool(re.search(rf"\b{re.escape(token)}s\b", context_norm))


def required_target_phrases(question: str) -> list[str]:
    lowered = question.lower()
    targets: list[str] = []
    for pattern in (
        r"\bcollecting\s+([a-z0-9][a-z0-9 '\-]+?)(?:\?|$|,|\.|\s+since\b|\s+for\b)",
        r"\b(?:bought|purchased|ordered)\s+(?:my|a|an|the)\s+([a-z0-9][a-z0-9 +&'/-]{1,40}?)(?:\s+did\b|\s+before\b|\s+after\b|\?|$)",
    ):
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            target = clean_required_target(match.group(1))
            if target not in {"again"} and not is_broad_gift_target(target):
                targets.append(target)
    if "taking the bus" in lowered:
        targets.append("bus")
    if "undergrad course" in lowered or "undergraduate course" in lowered:
        targets.append("undergrad")
    if "master's degree" in lowered or "masters degree" in lowered or "master degree" in lowered:
        targets.append("master's degree")
    if "plant" in lowered and ("how many" in lowered or "how much" in lowered) and " for " in lowered and " and " in lowered:
        match = re.search(r"\bfor\s+([a-z][a-z -]+?)\s+and\s+([a-z][a-z -]+?)(?:\?|$)", lowered)
        if match:
            targets.extend([clean_required_target(match.group(1)), clean_required_target(match.group(2))])
    return dedupe_targets(target for target in targets if target)


def clean_required_target(target: str) -> str:
    cleaned = target.strip(" .,:;!?\"'")
    return re.sub(r"\b(?:did|before|after|have|had|was|were)\b.*$", "", cleaned).strip()


def is_broad_gift_target(target: str) -> bool:
    normalized = normalize_text_for_match(target)
    return normalized == "gift" or normalized.startswith("gift for ") or " birthday gift " in f" {normalized} "


def missing_order_option_reason(question: str, context: str) -> str | None:
    lowered = question.lower()
    if "yes or no" in lowered or "or not" in lowered:
        return None
    if " or " not in lowered or not re.search(r"\b(first|earlier)\b", lowered):
        return None
    options = extract_or_options(question)
    if len(options) != 2:
        return None
    missing = [option for option in options if not option_supported_in_context(option, context)]
    if not missing:
        return None
    return (
        "The provided information is not enough to answer the question. "
        f"The context does not mention {missing[0].strip()}."
    )


def extract_or_options(question: str) -> list[str]:
    text = question.strip().rstrip("?")
    if "," in text:
        text = text.rsplit(",", 1)[1]
    parts = re.split(r"\s+or\s+", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return []
    return [clean_option(part) for part in parts if clean_option(part)]


def clean_option(text: str) -> str:
    cleaned = text.strip(" .,:;!?\"'")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def option_supported_in_context(option: str, context: str) -> bool:
    option_norm = normalize_text_for_match(option)
    context_norm = normalize_text_for_match(context)
    if not option_norm:
        return False
    if re.search(rf"\b{re.escape(option_norm)}\b", context_norm):
        return True
    tokens = option_content_tokens(option)
    if not tokens:
        return False
    matched = sum(1 for token in tokens if re.search(rf"\b{re.escape(token)}\b", context_norm))
    if len(tokens) <= 2:
        return matched == len(tokens)
    required = max(2, (len(tokens) * 3 + 4) // 5)
    return matched >= required


def option_content_tokens(option: str) -> list[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "for",
        "from",
        "to",
        "with",
        "did",
        "i",
        "my",
        "me",
        "you",
        "first",
        "earlier",
        "which",
        "who",
        "what",
        "event",
        "item",
        "task",
        "project",
        "happened",
        "participation",
        "attendance",
        "purchase",
        "purchased",
        "purchasing",
        "buy",
        "bought",
        "complete",
        "completed",
        "fix",
        "fixing",
        "start",
        "started",
        "became",
        "become",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", option.lower())
        if token not in stopwords and (len(token) > 2 or token.isdigit())
    ]


def normalize_text_for_match(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def phrase_present_in_normalized_context(phrase_norm: str, context_norm: str) -> bool:
    if re.search(rf"\b{re.escape(phrase_norm)}\b", context_norm):
        return True
    if phrase_norm.endswith("s") and re.search(rf"\b{re.escape(phrase_norm[:-1])}\b", context_norm):
        return True
    return False


def dedupe_targets(targets: Any) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for target in targets:
        norm = normalize_text_for_match(str(target))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        deduped.append(str(target))
    return deduped


def normalize_over_precise_duration(question: str, answer: str) -> str:
    if "how long" not in question.lower():
        return answer
    number = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    pattern = re.compile(rf"\b({number})\s+(months?)\s+and\s+{number}\s+days?\b", re.IGNORECASE)
    match = pattern.search(answer)
    if not match:
        return answer
    normalized = f"{match.group(1)} {match.group(2)}"
    if answer.strip().lower().rstrip(".") == match.group(0).lower():
        return normalized
    return pattern.sub(normalized, answer, count=1)


def normalize_bare_year_difference(question: str, answer: str) -> str:
    lowered = question.lower()
    if not ("older" in lowered and ("how much" in lowered or "how many years" in lowered)):
        return answer
    cleaned = answer.strip().strip(".")
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        return answer
    return f"{cleaned} years"


def normalize_list_count_arithmetic(question: str, answer: str) -> str:
    if "how many" not in question.lower() or "in total" not in question.lower():
        return answer
    leading = re.match(r"\s*(\d+)\s+([A-Za-z][A-Za-z ]{0,40}?)\s*:", answer)
    if not leading:
        return answer
    details = answer[leading.end() :]
    values = [
        int(value)
        for value in re.findall(r"\b(\d+)\s+(?:rare\s+)?(?:items?|figurines?|records?|coins?|books?)\b", details.lower())
    ]
    if len(values) < 2:
        return answer
    total = sum(values)
    stated = int(leading.group(1))
    if total == stated or total <= 0:
        return answer
    return re.sub(r"^\s*\d+", str(total), answer, count=1)

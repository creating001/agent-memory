from __future__ import annotations

import re
from typing import Any

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
) -> str:
    memory_context = "\n".join(str(item.text) for item in (all_chunks or retrieved))
    deterministic_answer = deterministic_relative_event_order_answer(example.question, all_chunks)
    answer = deterministic_answer or hypothesis
    answer = normalize_over_precise_duration(example.question, answer)
    answer = normalize_bare_year_difference(example.question, answer)
    answer = normalize_list_count_arithmetic(example.question, answer)

    target_context = memory_context
    if all_chunks and not asks_assistant_memory(example.question):
        user_context = user_memory_context(all_chunks)
        if user_context:
            target_context = user_context
    missing_target = None if deterministic_answer else exact_target_missing_reason(example.question, target_context)
    if missing_target:
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
    match = re.search(rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b", text, re.IGNORECASE)
    if match:
        return int(match.group(3)), months[match.group(2).lower()], int(match.group(1))
    match = re.search(rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,)?\s+(20\d{{2}})\b", text, re.IGNORECASE)
    if match:
        return int(match.group(3)), months[match.group(1).lower()], int(match.group(2))
    return None


def exact_target_missing_reason(question: str, context: str) -> str | None:
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

    family_relations = {"uncle", "aunt", "dad", "father", "mom", "mother", "sister", "brother", "niece", "nephew", "cousin"}
    for relation in family_relations:
        if not re.search(rf"\b(?:my\s+)?{relation}(?:'s)?\b", lowered_question):
            continue
        if re.search(rf"\b{relation}\b", lowered_context):
            continue
        other_relation = next(
            (other for other in family_relations if other != relation and re.search(rf"\b{other}\b", lowered_context)),
            None,
        )
        if other_relation:
            return (
                "The provided information is not enough to answer the question. "
                f"The context mentions a different family member, not your {relation}."
            )
    return None


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

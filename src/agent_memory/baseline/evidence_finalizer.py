from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from agent_memory.core.llm import extract_json_object
from agent_memory.core.schema import Example, RetrievedChunk


@dataclass(frozen=True)
class EvidenceFinalization:
    answer: str
    reason: str


_PLAN_OR_FUTURE = re.compile(
    r"\b("
    r"plan(?:ned|s|ning)?|future|would|could|might|will|going to|intend|intention|"
    r"thinking of|considering|looking to|want to|schedule|scheduled|not completed|"
    r"not confirmed|only planning|hypothetical"
    r")\b",
    re.IGNORECASE,
)
_DUPLICATE = re.compile(r"\b(duplicate|repeated mention|same event|same item|same object)\b", re.IGNORECASE)
_NEGATED_PLAN_OR_HYPOTHETICAL = re.compile(
    r"\bnot\b.{0,80}\b(plan|planned|planning|discussion|hypothetical|suggestion|future)\b",
    re.IGNORECASE,
)
_CURRENT_POSSESSION = re.compile(
    r"\b(still own|still owns|still owned|still have|still has|still in (?:my|their|his|her)?\s*possession|"
    r"currently own|currently owns|current ownership|haven't sold|has not sold|not sold yet|still thinking about selling)\b",
    re.IGNORECASE,
)
_DISTINCT_EVENT_OR_ITEM = re.compile(
    r"\b(distinct|separate|different|another|second|third|new)\b.{0,60}\b(event|item|object|instance|time|injury|trip|hike|purchase)\b|"
    r"\b(event|item|object|instance|time|injury|trip|hike|purchase)\b.{0,60}\b(distinct|separate|different|another|second|third|new)\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_NUMBER = re.compile(r"(?<![A-Za-z])([0-9][0-9,]*(?:\.[0-9]+)?)(?![A-Za-z])")
_DATE_ISO = re.compile(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b")
_MONTH_DATE = re.compile(
    r"\b("
    r"january|february|march|april|may|june|july|august|september|october|november|december"
    r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,\s*(20\d{2}))?\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "did",
    "different",
    "do",
    "does",
    "for",
    "from",
    "have",
    "how",
    "i",
    "in",
    "is",
    "many",
    "much",
    "my",
    "number",
    "of",
    "on",
    "or",
    "since",
    "the",
    "times",
    "to",
    "total",
    "what",
    "which",
    "with",
}


def finalize_answer_from_evidence(
    example: Example,
    evidence_response: str | None,
    retrieved: list[RetrievedChunk],
    hypothesis: str | None = None,
) -> EvidenceFinalization | None:
    """Repair mechanical inconsistencies between evidence JSON and final answer.

    The model is still responsible for retrieval and evidence extraction.  This
    layer only fires when a structured extraction already exposes a narrow,
    deterministic correction: additive arithmetic, duplicated counted items, or
    planned/hypothetical evidence accidentally counted as completed evidence.
    """

    lowered_question = example.question.lower()
    if _is_order_question(lowered_question):
        order = _finalize_ordered_answer(hypothesis)
        if order:
            return order

    if not evidence_response:
        return None

    evidence = extract_json_object(evidence_response)
    if not evidence:
        return None

    if evidence.get("sufficient") is False:
        insufficient = _finalize_insufficient_answer(evidence, hypothesis)
        if insufficient:
            return insufficient
        duration = _finalize_duration_from_evidence(lowered_question, raw_items=_raw_evidence_items(evidence), hypothesis=hypothesis)
        if duration:
            return duration
        return None

    if not (_is_count_question(lowered_question) or _is_sum_question(lowered_question)):
        return None

    raw_items = _raw_evidence_items(evidence)
    if not raw_items:
        return None
    included = [_normalize_item(item) for item in raw_items if _item_is_included(item)]

    duration = _finalize_duration_from_evidence(lowered_question, raw_items=raw_items, hypothesis=hypothesis)
    if duration:
        return duration

    if _is_sum_question(lowered_question) and _is_additive_sum_question(lowered_question):
        conditional_total = _finalize_baseline_delta_quantity(
            lowered_question,
            [_normalize_item(item) for item in raw_items],
            hypothesis,
        )
        if conditional_total:
            return conditional_total
        filtered = [item for item in included if not _item_should_be_filtered(item, lowered_question)]
        filtered = _dedupe_items(filtered, distinct_mode=_asks_distinct(lowered_question))
        numeric_items = _numeric_items_for_sum(
            filtered,
            lowered_question=lowered_question,
            money=_is_money_question(lowered_question),
        )
        numeric_items.extend(
            _repair_missing_numeric_items(
                raw_items,
                retrieved,
                lowered_question=lowered_question,
                money=_is_money_question(lowered_question),
            )
        )
        numeric_items = _dedupe_numeric_items(numeric_items)
        if len(numeric_items) < 2:
            return None
        total = sum(value for _, value, _ in numeric_items)
        unit = _sum_unit(lowered_question, numeric_items)
        if unit not in {"$", "views"}:
            return None
        if not _numeric_answer_disagrees(hypothesis, total):
            return None
        answer = _format_sum_answer(total, unit, numeric_items)
        return EvidenceFinalization(answer=answer, reason="evidence_arithmetic_consistency")

    if _is_count_question(lowered_question):
        distinct = _finalize_distinct_value_count(lowered_question, included, hypothesis)
        if distinct:
            return distinct
        conditional_total = _finalize_baseline_delta_quantity(
            lowered_question,
            [_normalize_item(item) for item in raw_items],
            hypothesis,
        )
        if conditional_total:
            return conditional_total
        if _is_duration_count_question(lowered_question):
            return None
        if not _count_correction_needed(included, lowered_question):
            return None
        filtered = [item for item in included if not _item_should_be_filtered(item, lowered_question)]
        filtered = _dedupe_items(filtered, distinct_mode=_asks_distinct(lowered_question))
        if not filtered:
            return None
        expected = Decimal(len(filtered))
        if not _count_answer_should_be_reduced(hypothesis, expected):
            return None
        answer = _format_count_answer(len(filtered), filtered)
        return EvidenceFinalization(answer=answer, reason="evidence_count_consistency")

    return None


def _raw_evidence_items(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    items = evidence.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    counted = evidence.get("counted_items")
    if isinstance(counted, list):
        return [item for item in counted if isinstance(item, dict)]
    return []


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "canonical_item": str(item.get("canonical_item") or item.get("item") or "").strip(),
        "date": str(item.get("date") or "").strip(),
        "evidence": str(item.get("evidence") or item.get("quote") or "").strip(),
        "value": item.get("value"),
        "reason": str(item.get("reason") or "").strip(),
    }


def _item_is_included(item: dict[str, Any]) -> bool:
    value = item.get("include")
    if value is None and "counted_items" not in item:
        return True
    return bool(value)


def _item_should_be_filtered(item: dict[str, Any], lowered_question: str) -> bool:
    if _question_allows_planned_items(lowered_question):
        return False
    reason = str(item.get("reason") or "")
    evidence = str(item.get("evidence") or "")
    if _CURRENT_POSSESSION.search(f"{reason} {evidence}"):
        return False
    if _NEGATED_PLAN_OR_HYPOTHETICAL.search(reason):
        return False
    if _PLAN_OR_FUTURE.search(reason):
        return True
    return False


def _dedupe_items(items: list[dict[str, Any]], *, distinct_mode: bool) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        key = _item_key(item, distinct_mode=distinct_mode)
        if key in seen:
            if _item_explicitly_distinct(item):
                key = f"{key}#{len(output)}"
            else:
                continue
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _item_key(item: dict[str, Any], *, distinct_mode: bool) -> str:
    value = _clean_key(str(item.get("value") or ""))
    canonical = _clean_key(str(item.get("canonical_item") or ""))
    evidence = _clean_key(str(item.get("evidence") or ""))
    if canonical:
        semantic = _semantic_item_key(canonical)
        return semantic or canonical
    if distinct_mode and value:
        return value
    return evidence[:160]


def _semantic_item_key(cleaned: str) -> str:
    tokens = [
        token
        for token in cleaned.split()
        if token not in {"group", "course", "class", "project", "projects", "the", "for"}
    ]
    if len(tokens) >= 2:
        return " ".join(tokens)
    return ""


def _item_explicitly_distinct(item: dict[str, Any]) -> bool:
    return bool(
        _DISTINCT_EVENT_OR_ITEM.search(
            f"{item.get('canonical_item', '')} {item.get('evidence', '')} {item.get('reason', '')}"
        )
    )


def _numeric_items_for_sum(
    items: list[dict[str, Any]],
    *,
    lowered_question: str,
    money: bool,
) -> list[tuple[str, Decimal, str]]:
    output: list[tuple[str, Decimal, str]] = []
    for item in items:
        value = _extract_numeric_value(item.get("value"), money=money)
        if value is None:
            value = _extract_numeric_value(item.get("evidence"), money=money)
        if value is None:
            continue
        label = str(item.get("canonical_item") or item.get("evidence") or "item").strip()
        output.append((label, value, str(item.get("evidence") or "")))
    return output


def _repair_missing_numeric_items(
    raw_items: list[dict[str, Any]],
    retrieved: list[RetrievedChunk],
    *,
    lowered_question: str,
    money: bool,
) -> list[tuple[str, Decimal, str]]:
    repaired: list[tuple[str, Decimal, str]] = []
    if not money:
        return repaired
    question_terms = _content_terms(lowered_question)
    for raw_item in raw_items:
        item = _normalize_item(raw_item)
        if _item_is_included(raw_item):
            continue
        if _extract_numeric_value(item.get("value"), money=money) is not None:
            continue
        reason = item.get("reason", "")
        if not re.search(r"\b(cost|price|amount|value|not specified|missing)\b", reason, re.IGNORECASE):
            continue
        item_terms = _content_terms(item.get("canonical_item", ""))
        if not item_terms:
            continue
        matches = []
        for chunk in retrieved:
            text = chunk.text
            lowered = text.lower()
            if not any(term in lowered for term in item_terms):
                continue
            if question_terms and not any(term in lowered for term in question_terms):
                continue
            for value in _extract_money_values(text):
                matches.append((value, text))
        unique_values = sorted({value for value, _ in matches})
        if len(unique_values) != 1:
            continue
        evidence_text = next(text for value, text in matches if value == unique_values[0])
        repaired.append((item["canonical_item"], unique_values[0], evidence_text))
    return repaired


def _dedupe_numeric_items(items: list[tuple[str, Decimal, str]]) -> list[tuple[str, Decimal, str]]:
    seen: set[tuple[str, str]] = set()
    output: list[tuple[str, Decimal, str]] = []
    for label, value, evidence in items:
        key = (_clean_key(label), str(value))
        if key in seen:
            continue
        seen.add(key)
        output.append((label, value, evidence))
    return output


def _extract_numeric_value(value: Any, *, money: bool) -> Decimal | None:
    if value is None:
        return None
    text = str(value)
    matches = _extract_money_values(text) if money else _extract_number_values(text)
    if not matches:
        return None
    return matches[0]


def _extract_money_values(text: str) -> list[Decimal]:
    return [_decimal(match.group(1)) for match in _MONEY.finditer(text) if _decimal(match.group(1)) is not None]


def _extract_number_values(text: str) -> list[Decimal]:
    return [_decimal(match.group(1)) for match in _NUMBER.finditer(text) if _decimal(match.group(1)) is not None]


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except (InvalidOperation, AttributeError):
        return None


def _is_sum_question(lowered_question: str) -> bool:
    return bool(
        re.search(r"\b(total|sum|altogether|combined)\b", lowered_question)
        or re.search(r"\bhow much\b.*\b(spend|spent|cost|pay|paid|money|expense)", lowered_question)
    )


def _is_additive_sum_question(lowered_question: str) -> bool:
    if re.search(r"\b(difference|more than|less than|how much more|how much less|average|percentage|percent)\b", lowered_question):
        return False
    return bool(
        re.search(r"\b(total|sum|altogether|combined)\b", lowered_question)
        or re.search(r"\bhow much\b.*\b(spend|spent|cost|pay|paid|money|expense)", lowered_question)
    )


def _is_money_question(lowered_question: str) -> bool:
    return bool(re.search(r"\b(money|spend|spent|cost|paid|pay|expense|expenses|price|dollar)\b", lowered_question))


def _is_count_question(lowered_question: str) -> bool:
    return bool(re.search(r"\bhow many\b|\bnumber of\b", lowered_question))


def _is_duration_count_question(lowered_question: str) -> bool:
    return bool(re.search(r"\bhow many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", lowered_question))


def _asks_distinct(lowered_question: str) -> bool:
    return bool(re.search(r"\b(different|distinct|unique)\b", lowered_question))


def _question_allows_planned_items(lowered_question: str) -> bool:
    return bool(
        re.search(
            r"\b(plan|planned|planning|intend|intention|schedule|scheduled|will|going to|would|could|might)\b",
            lowered_question,
        )
    )


def _count_correction_needed(items: list[dict[str, Any]], lowered_question: str) -> bool:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if _item_should_be_filtered(item, lowered_question):
            return True
        key = _item_key(item, distinct_mode=False)
        if key:
            grouped.setdefault(key, []).append(item)
    for duplicates in grouped.values():
        if len(duplicates) > 1 and all(_is_unit_count_value(item.get("value")) for item in duplicates):
            return True
    return False


def _is_unit_count_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "1", "one"}


def _numeric_answer_disagrees(hypothesis: str | None, expected: Decimal) -> bool:
    if not hypothesis:
        return True
    values = _extract_money_values(hypothesis) or _extract_number_values(hypothesis)
    if not values:
        return True
    return values[0] != expected


def _count_answer_should_be_reduced(hypothesis: str | None, expected: Decimal) -> bool:
    if not hypothesis:
        return False
    values = _extract_number_values(hypothesis)
    if not values:
        return False
    return values[0] > expected


def _finalize_insufficient_answer(
    evidence: dict[str, Any],
    hypothesis: str | None,
) -> EvidenceFinalization | None:
    if not hypothesis or not re.search(r"\bnot enough|insufficient|cannot determine|can't determine\b", hypothesis, re.IGNORECASE):
        return None
    if len(hypothesis.split()) > 10:
        return None
    missing_info = str(evidence.get("missing_info") or "").strip()
    if not missing_info:
        return None
    return EvidenceFinalization(
        answer=f"The provided information is not enough to answer the question: {missing_info}",
        reason="evidence_missing_target_detail",
    )


def _finalize_distinct_value_count(
    lowered_question: str,
    items: list[dict[str, Any]],
    hypothesis: str | None,
) -> EvidenceFinalization | None:
    if not (_asks_distinct(lowered_question) or re.search(r"\btypes? of\b", lowered_question)):
        return None
    values: list[str] = []
    for item in items:
        if _item_should_be_filtered(item, lowered_question):
            continue
        values.extend(_split_distinct_values(item.get("value")))
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = _clean_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    if len(output) < 2:
        return None
    expected = Decimal(len(output))
    if not _numeric_answer_disagrees(hypothesis, expected):
        return None
    return EvidenceFinalization(
        answer=f"{len(output)}: {', '.join(output[:8])}",
        reason="evidence_distinct_value_consistency",
    )


def _split_distinct_values(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or _extract_number_values(text):
        return []
    parts = re.split(r"\s*(?:,|/|\band\b|\bor\b)\s*", text, flags=re.IGNORECASE)
    output = []
    for part in parts:
        cleaned = part.strip(" .;:()[]{}\"'")
        if len(cleaned) >= 2:
            output.append(cleaned)
    return output


def _finalize_baseline_delta_quantity(
    lowered_question: str,
    items: list[dict[str, Any]],
    hypothesis: str | None,
) -> EvidenceFinalization | None:
    if not re.search(r"\b(how many|how much)\b", lowered_question):
        return None
    baseline: Decimal | None = None
    delta: Decimal | None = None
    unit = ""
    for item in items:
        value = _extract_numeric_value(item.get("value"), money=_is_money_question(lowered_question))
        if value is None:
            value = _extract_numeric_value(item.get("evidence"), money=_is_money_question(lowered_question))
        if value is None:
            continue
        text = f"{item.get('canonical_item', '')} {item.get('evidence', '')} {item.get('reason', '')}".lower()
        if re.search(r"\b(usual|usually|typical|baseline|regular|normally)\b", text):
            baseline = value
        if re.search(r"\b(increase|increased|additional|extra|more|up by|go up by)\b", text):
            delta = value
        if re.search(r"\bhours?\b", text):
            unit = "hours"
    if baseline is None or delta is None:
        return None
    total = baseline + delta
    if not _numeric_answer_disagrees(hypothesis, total):
        return None
    unit = unit or _sum_unit(lowered_question, [])
    return EvidenceFinalization(
        answer=_format_value(total, unit),
        reason="evidence_baseline_delta_consistency",
    )


def _finalize_duration_from_evidence(
    lowered_question: str,
    *,
    raw_items: list[dict[str, Any]],
    hypothesis: str | None,
) -> EvidenceFinalization | None:
    unit = _duration_unit(lowered_question)
    if unit != "days":
        return None
    if not _endpoint_duration_question(lowered_question):
        return None
    dates = _evidence_endpoint_dates(raw_items)
    if len(dates) < 2:
        calc_value = _single_calculation_number(raw_items)
        if calc_value is None:
            return None
        if not _duration_answer_needs_normalization(hypothesis, calc_value):
            return None
        return EvidenceFinalization(
            answer=f"{int(calc_value)} days",
            reason="evidence_duration_format_consistency",
        )
    start = min(dates)
    end = max(dates)
    days = Decimal((end - start).days)
    if days <= 0:
        return None
    if not _duration_endpoint_disagrees(hypothesis, days):
        return None
    return EvidenceFinalization(
        answer=f"{int(days)} days",
        reason="evidence_duration_endpoint_consistency",
    )


def _duration_unit(lowered_question: str) -> str:
    match = re.search(r"\bhow many\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", lowered_question)
    if not match:
        return ""
    unit = match.group(1).rstrip("s")
    return f"{unit}s" if unit != "day" else "days"


def _endpoint_duration_question(lowered_question: str) -> bool:
    return bool(
        re.search(
            r"\b(passed between|between the day|did it take|days? ago|days? before|days? after)\b",
            lowered_question,
        )
    )


def _duration_endpoint_disagrees(hypothesis: str | None, expected_days: Decimal) -> bool:
    if not hypothesis:
        return True
    if re.search(r"\bnot enough|insufficient|cannot determine|can't determine\b", hypothesis, re.IGNORECASE):
        return True
    values = _extract_number_values(hypothesis)
    if not values:
        return True
    return abs(values[0] - expected_days) > 1


def _evidence_endpoint_dates(raw_items: list[dict[str, Any]]) -> list[date]:
    dates: list[date] = []
    for raw in raw_items:
        item = _normalize_item(raw)
        candidates = [
            str(item.get("value") or ""),
            str(item.get("evidence") or ""),
            str(item.get("date") or ""),
        ]
        parsed = None
        for candidate in candidates:
            parsed = _first_date(candidate)
            if parsed:
                break
        if parsed:
            dates.append(parsed)
    return dates


def _first_date(text: str) -> date | None:
    iso = _DATE_ISO.search(text)
    if iso:
        return _date_from_parts(iso.group(1), iso.group(2), iso.group(3))
    month = _MONTH_DATE.search(text)
    if month:
        year = month.group(3) or "2023"
        return _date_from_parts(year, str(_month_number(month.group(1))), month.group(2))
    return None


def _date_from_parts(year: str, month: str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _month_number(name: str) -> int:
    return datetime.strptime(name[:3].title(), "%b").month


def _single_calculation_number(raw_items: list[dict[str, Any]]) -> Decimal | None:
    return None


def _duration_answer_needs_normalization(hypothesis: str | None, value: Decimal) -> bool:
    if not hypothesis:
        return True
    if _numeric_answer_disagrees(hypothesis, value):
        return False
    normalized = hypothesis.strip().lower()
    return not re.fullmatch(rf"{int(value)}\s+days?\.?", normalized)


def _is_order_question(lowered_question: str) -> bool:
    return bool(re.search(r"\border of\b|\bwhat.*\bfirst\b|\bwhich.*\bfirst\b", lowered_question))


def _finalize_ordered_answer(hypothesis: str | None) -> EvidenceFinalization | None:
    if not hypothesis:
        return None
    segments = _dated_answer_segments(hypothesis)
    if len(segments) < 3:
        return None
    dates = [item[0] for item in segments]
    if dates == sorted(dates):
        return None
    ordered = sorted(segments, key=lambda item: item[0])
    answer = "; ".join(f"{index}. {segment}" for index, (_, segment) in enumerate(ordered, start=1))
    return EvidenceFinalization(answer=answer, reason="answer_chronological_order_consistency")


def _dated_answer_segments(text: str) -> list[tuple[date, str]]:
    raw_segments = [segment.strip(" .;\n") for segment in re.split(r"(?:^|\n|\s)\d+[.)]\s+", text) if segment.strip()]
    if len(raw_segments) < 3:
        raw_segments = [segment.strip(" .;\n") for segment in re.split(r";\s*", text) if segment.strip()]
    segments: list[tuple[date, str]] = []
    for segment in raw_segments:
        parsed = _first_date(segment)
        if parsed:
            segments.append((parsed, segment))
    return segments


def _sum_unit(lowered_question: str, items: list[tuple[str, Decimal, str]]) -> str:
    if _is_money_question(lowered_question):
        return "$"
    if "view" in lowered_question:
        return "views"
    return ""


def _format_sum_answer(total: Decimal, unit: str, items: list[tuple[str, Decimal, str]]) -> str:
    operands = ", ".join(f"{_format_value(value, unit)} from {label}" for label, value, _ in items[:5])
    total_text = _format_value(total, unit)
    if operands:
        return f"{total_text} ({operands})"
    return total_text


def _format_count_answer(count: int, items: list[dict[str, Any]]) -> str:
    names = [str(item.get("canonical_item") or item.get("value") or "").strip() for item in items]
    names = [name for name in names if name]
    if names:
        return f"{count}: {', '.join(names[:8])}"
    return str(count)


def _format_value(value: Decimal, unit: str) -> str:
    number = f"{value:,.2f}" if value != value.to_integral() else f"{int(value):,}"
    if unit == "$":
        return f"${number}"
    if unit:
        return f"{number} {unit}"
    return number


def _content_terms(text: str) -> list[str]:
    terms = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9]{2,}", text.lower()):
        if token in _STOPWORDS:
            continue
        terms.append(token)
    return terms[:12]


def _clean_key(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))

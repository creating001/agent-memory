from __future__ import annotations

from calendar import monthrange
from datetime import datetime, timedelta
import re

from agent_memory.core.schema import Chunk, Example, Turn


def build_chunks(
    example: Example,
    *,
    chunk_unit: str,
    relative_time_annotations: bool = False,
    turn_pair_chunks: bool = False,
    turn_pair_max_chars: int = 2500,
) -> list[Chunk]:
    if chunk_unit == "turn":
        chunks = _turn_chunks(example.turns, relative_time_annotations=relative_time_annotations)
        if turn_pair_chunks:
            chunks.extend(_turn_pair_chunks(example.turns, max_chars=turn_pair_max_chars))
        return chunks
    raise ValueError(f"Strong memory baseline only supports turn chunking, got: {chunk_unit}")


def _turn_chunks(turns: list[Turn], *, relative_time_annotations: bool) -> list[Chunk]:
    chunks = []
    for index, turn in enumerate(turns):
        text = f"{turn.role}: {turn.content}"
        if relative_time_annotations:
            annotation = _relative_time_annotation(turn.content, turn.date)
            if annotation:
                text = f"{text}\n[Resolved relative time: {annotation}]"
        chunks.append(Chunk(chunk_id=f"turn-{index:05d}", text=text, date=turn.date, session_id=turn.session_id))
    return chunks


def _turn_pair_chunks(turns: list[Turn], *, max_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, turn in enumerate(turns[:-1]):
        following = turns[index + 1]
        if turn.role != "user" or following.role != "assistant":
            continue
        if turn.session_id and following.session_id and turn.session_id != following.session_id:
            continue
        text = _truncate_turn_pair_text(
            user_text=f"user: {turn.content}",
            assistant_text=f"assistant: {following.content}",
            max_chars=max_chars,
        )
        chunks.append(
            Chunk(
                chunk_id=f"turn-pair-{index:05d}-{index + 1:05d}",
                text=text,
                date=turn.date,
                session_id=turn.session_id or following.session_id,
            )
        )
    return chunks


def _truncate_turn_pair_text(*, user_text: str, assistant_text: str, max_chars: int) -> str:
    text = f"{user_text}\n{assistant_text}"
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    separator = "\n"
    budget = max_chars - len(separator)
    if budget <= 0:
        return text[:max_chars]
    user_budget = max(64, budget // 2)
    assistant_budget = max(64, budget - user_budget)
    return f"{_truncate_middle(user_text, user_budget)}{separator}{_truncate_middle(assistant_text, assistant_budget)}"


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    marker = " ... "
    if max_chars <= len(marker) + 8:
        return text[:max_chars]
    head = (max_chars - len(marker)) // 2
    tail = max_chars - len(marker) - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def relative_time_annotation(text: str, session_date: str) -> str:
    return _relative_time_annotation(text, session_date)


def _relative_time_annotation(text: str, session_date: str) -> str:
    base_date = _parse_session_date(session_date)
    if base_date is None:
        return ""

    lowered = text.lower()
    items: list[tuple[str, datetime | tuple[datetime, datetime]]] = []

    phrase_offsets = {
        "day before yesterday": -2,
        "yesterday": -1,
        "tomorrow": 1,
        "day after tomorrow": 2,
    }
    for phrase, offset in phrase_offsets.items():
        if re.search(rf"\b{re.escape(phrase)}\b", lowered):
            items.append((phrase, base_date + timedelta(days=offset)))

    amount_offsets = {
        "days": "day",
        "day": "day",
        "weeks": "week",
        "week": "week",
        "months": "month",
        "month": "month",
        "years": "year",
        "year": "year",
    }
    for match in re.finditer(
        r"\b(?P<amount>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<unit>days?|weeks?|months?|years?)\s+ago\b",
        lowered,
    ):
        amount = _number_value(match.group("amount"))
        unit = amount_offsets[match.group("unit")]
        if amount is not None:
            items.append((match.group(0), _shift_date(base_date, unit, -amount)))

    for phrase, unit, amount in (
        ("last week", "week", -1),
        ("next week", "week", 1),
        ("last month", "month", -1),
        ("next month", "month", 1),
        ("last year", "year", -1),
        ("next year", "year", 1),
    ):
        if re.search(rf"\b{phrase}\b", lowered):
            items.append((phrase, _shift_date(base_date, unit, amount)))

    if "last weekend" in lowered:
        items.append(("last weekend", _previous_weekend(base_date)))
    if "next weekend" in lowered:
        items.append(("next weekend", _next_weekend(base_date)))

    weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    for day_name, day_index in weekdays.items():
        if re.search(rf"\blast {day_name}\b", lowered):
            items.append((f"last {day_name}", _relative_weekday(base_date, day_index, direction=-1)))
        if re.search(rf"\bnext {day_name}\b", lowered):
            items.append((f"next {day_name}", _relative_weekday(base_date, day_index, direction=1)))

    if not items:
        return ""

    unique: dict[str, str] = {}
    for phrase, value in items:
        if isinstance(value, tuple):
            rendered = f"{_format_date(value[0])} to {_format_date(value[1])}"
        else:
            rendered = _format_date(value)
        unique.setdefault(phrase, rendered)
    return "; ".join(f"{phrase} = {value}" for phrase, value in unique.items())


def _parse_session_date(value: str) -> datetime | None:
    cleaned = " ".join(str(value or "").split())
    cleaned = re.sub(r"\bat\b.*$", "", cleaned).strip()
    cleaned = cleaned.replace(",", "")
    day_month = re.search(r"\b\d{1,2}\s+[A-Za-z]+\s+20\d{2}\b", cleaned)
    if day_month:
        cleaned = day_month.group(0)
    else:
        month_day = re.search(r"\b[A-Za-z]+\s+\d{1,2}\s+20\d{2}\b", cleaned)
        if month_day:
            cleaned = month_day.group(0)
        else:
            year_month_day = re.search(r"\b20\d{2}[-/]\d{1,2}[-/]\d{1,2}\b", cleaned)
            if year_month_day:
                cleaned = year_month_day.group(0)
    for date_format in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(cleaned, date_format)
        except ValueError:
            continue
    return None


def _number_value(value: str) -> int | None:
    words = {
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
    if value.isdigit():
        return int(value)
    return words.get(value)


def _shift_date(base_date: datetime, unit: str, amount: int) -> datetime:
    if unit == "day":
        return base_date + timedelta(days=amount)
    if unit == "week":
        return base_date + timedelta(days=7 * amount)
    if unit == "month":
        month_index = base_date.month - 1 + amount
        year = base_date.year + month_index // 12
        month = month_index % 12 + 1
        day = min(base_date.day, monthrange(year, month)[1])
        return base_date.replace(year=year, month=month, day=day)
    if unit == "year":
        year = base_date.year + amount
        day = min(base_date.day, monthrange(year, base_date.month)[1])
        return base_date.replace(year=year, day=day)
    return base_date


def _previous_weekend(base_date: datetime) -> tuple[datetime, datetime]:
    days_since_sunday = (base_date.weekday() - 6) % 7 or 7
    sunday = base_date - timedelta(days=days_since_sunday)
    return sunday - timedelta(days=1), sunday


def _next_weekend(base_date: datetime) -> tuple[datetime, datetime]:
    days_until_saturday = (5 - base_date.weekday()) % 7 or 7
    saturday = base_date + timedelta(days=days_until_saturday)
    return saturday, saturday + timedelta(days=1)


def _relative_weekday(base_date: datetime, weekday: int, *, direction: int) -> datetime:
    if direction < 0:
        delta = (base_date.weekday() - weekday) % 7 or 7
        return base_date - timedelta(days=delta)
    delta = (weekday - base_date.weekday()) % 7 or 7
    return base_date + timedelta(days=delta)


def _format_date(value: datetime) -> str:
    return value.strftime("%d %B %Y").lstrip("0")

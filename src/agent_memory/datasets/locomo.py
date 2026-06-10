from __future__ import annotations

from typing import Any

from agent_memory.core.schema import Example, Turn


LOCOMO_CATEGORY_NAMES = {
    "1": "multi-hop",
    "2": "temporal",
    "3": "open-domain",
    "4": "single-hop",
    "5": "adversarial",
}


def locomo_category_name(category: Any) -> str:
    return LOCOMO_CATEGORY_NAMES.get(str(category), f"unknown-category-{category}")


def load_locomo_records(records: list[dict[str, Any]]) -> list[Example]:
    examples: list[Example] = []
    for record_index, record in enumerate(records):
        conversation_id = str(record.get("sample_id") or f"locomo_{record_index}")
        conversation = record.get("conversation") or {}
        speaker_a = str(conversation.get("speaker_a") or "speaker_a")
        speaker_b = str(conversation.get("speaker_b") or "speaker_b")
        turns = _conversation_turns(conversation, speaker_a, speaker_b)
        session_metadata = _session_metadata(conversation)

        for qa_index, qa in enumerate(record.get("qa") or [], start=1):
            if not isinstance(qa, dict):
                continue
            original_qa_index = _qa_index(qa, qa_index)
            category = qa.get("category")
            sample_id = f"{conversation_id}_qa_{original_qa_index:03d}"
            answer = qa.get("answer")
            if str(category) == "5" and not answer:
                answer = "Not mentioned in the conversation"
            examples.append(
                Example(
                    sample_id=sample_id,
                    memory_id=conversation_id,
                    dataset="locomo",
                    question=str(qa.get("question") or ""),
                    question_date=None,
                    answer=answer,
                    question_type=locomo_category_name(category),
                    turns=turns,
                    metadata={
                        **qa,
                        "conversation_id": conversation_id,
                        "category": category,
                        "qa_index": original_qa_index,
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        **session_metadata,
                    },
                )
            )
    return examples


def _qa_index(qa: dict[str, Any], fallback: int) -> int:
    try:
        value = int(qa.get("qa_index") or fallback)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0 else fallback


def _session_metadata(conversation: dict[str, Any]) -> dict[str, Any]:
    return {"locomo_session_dates": _session_dates(conversation)}


def _session_dates(conversation: dict[str, Any]) -> dict[str, str]:
    dates: dict[str, str] = {}
    for key, value in conversation.items():
        if not key.startswith("session_") or not key.endswith("_date_time"):
            continue
        session_id = key.removesuffix("_date_time")
        dates[session_id] = str(value or "")
    return dates


def _conversation_turns(conversation: dict[str, Any], speaker_a: str, speaker_b: str) -> list[Turn]:
    turns: list[Turn] = []
    session_numbers = sorted(
        int(key.split("_")[1])
        for key, value in conversation.items()
        if key.startswith("session_")
        and key.count("_") == 1
        and key.split("_")[1].isdigit()
        and isinstance(value, list)
    )

    for session_number in session_numbers:
        session_id = f"session_{session_number}"
        date = str(conversation.get(f"{session_id}_date_time") or "")
        for row in conversation.get(session_id, []):
            if not isinstance(row, dict):
                continue
            speaker = str(row.get("speaker") or "")
            text = " ".join(str(row.get("text") or "").split())
            caption = row.get("blip_caption")
            if caption:
                text = f"{text} [Image caption: {caption}]".strip()
            if not text:
                continue
            role = "user" if speaker == speaker_a else "assistant" if speaker == speaker_b else speaker or "unknown"
            turns.append(Turn(role=role, content=f"{speaker}: {text}", date=date, session_id=session_id))
    return turns

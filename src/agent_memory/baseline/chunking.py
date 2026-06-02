from __future__ import annotations

from agent_memory.core.schema import Chunk, Example, Turn


def build_chunks(example: Example, *, chunk_unit: str) -> list[Chunk]:
    if chunk_unit == "turn":
        return _turn_chunks(example.turns)
    raise ValueError(f"LTS baseline only supports turn chunking, got: {chunk_unit}")


def _turn_chunks(turns: list[Turn]) -> list[Chunk]:
    chunks = []
    for index, turn in enumerate(turns):
        text = f"{turn.role}: {turn.content}"
        chunks.append(Chunk(chunk_id=f"turn-{index:05d}", text=text, date=turn.date, session_id=turn.session_id))
    return chunks

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

import numpy as np

from agent_memory.core.config import get_config, load_config
from agent_memory.core.embedding import EmbeddingClient
from agent_memory.core.io import append_jsonl, completed_ids, load_env_file
from agent_memory.core.llm import ChatClient, extract_json_object
from agent_memory.core.schema import Chunk, Example, RetrievedChunk, Turn
from agent_memory.datasets import load_examples


@dataclass(frozen=True)
class RuntimeClients:
    answer: ChatClient
    embedding: EmbeddingClient | None
    config: dict[str, Any]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("src/agent_memory/configs/base.yaml"))
    parser.add_argument("--dataset", choices=("auto", "longmemeval", "locomo"), default="auto")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--include-question-type", action="append", default=[])
    parser.add_argument("--exclude-question-type", action="append", default=[])
    parser.add_argument("--max-context-chars", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")


def load_filtered_examples(args: argparse.Namespace) -> list[Example]:
    examples = load_examples(args.data, dataset=args.dataset)
    examples = filter_examples(
        examples,
        include_question_types=args.include_question_type,
        exclude_question_types=args.exclude_question_type,
    )
    if args.start:
        examples = examples[args.start :]
    if args.limit:
        examples = examples[: args.limit]
    return examples


def filter_examples(
    examples: list[Example],
    *,
    include_question_types: list[str],
    exclude_question_types: list[str],
) -> list[Example]:
    include = normalized_type_set(include_question_types)
    exclude = normalized_type_set(exclude_question_types)
    if not include and not exclude:
        return examples
    result = []
    for example in examples:
        question_type = normalize_question_type(example.question_type)
        if include and question_type not in include:
            continue
        if exclude and question_type in exclude:
            continue
        result.append(example)
    return result


def normalized_type_set(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        for item in str(value).split(","):
            normalized = normalize_question_type(item)
            if normalized:
                result.add(normalized)
    return result


def normalize_question_type(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", "-").split())


def make_clients(config_path: Path, *, include_embedding: bool) -> RuntimeClients:
    config = load_config(config_path)
    load_env_file(str(get_config(config, "paths.env_file", ".env")))

    answer_cfg = config["answer"]
    answer_api_key = os.environ.get(str(answer_cfg.get("api_key_env", "LOCAL_LLM_API_KEY")), "")
    if not answer_api_key:
        raise ValueError("Missing answer model API key environment variable.")

    embedding = None
    if include_embedding:
        embedding_cfg = config["embedding"]
        embedding = EmbeddingClient(
            model=str(embedding_cfg["name"]),
            base_url=str(embedding_cfg["base_url"]),
            api_key=str(embedding_cfg.get("api_key", "EMPTY")),
            batch_size=int(embedding_cfg.get("batch_size", 64)),
            normalize=bool(embedding_cfg.get("normalize", True)),
            query_instruction=str(embedding_cfg.get("query_instruction", "")),
            max_input_bytes=int(embedding_cfg.get("max_input_bytes", 0)),
        )

    return RuntimeClients(
        answer=ChatClient(
            api_key=answer_api_key,
            base_url=str(answer_cfg.get("base_url") or ""),
            timeout_seconds=float(answer_cfg.get("timeout_seconds", 120)),
            max_retries=int(answer_cfg.get("max_retries", 2)),
            default_seed=optional_int(answer_cfg.get("seed")),
            default_top_p=optional_float(answer_cfg.get("top_p")),
        ),
        embedding=embedding,
        config=config,
    )


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def answer_model_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["answer"]


def call_answer_model(
    client: ChatClient,
    config: dict[str, Any],
    messages: list[dict[str, str]],
) -> tuple[str, str, int]:
    answer_cfg = answer_model_config(config)
    result = client.complete(
        model=str(answer_cfg["name"]),
        messages=messages,
        temperature=float(answer_cfg.get("temperature", 0.0)),
        max_tokens=int(answer_cfg.get("final_max_tokens", min(int(answer_cfg.get("max_tokens", 8192)), 1024))),
        thinking=str(answer_cfg.get("thinking", "default")),
        response_format={"type": "json_object"},
    )
    return parse_answer(result.content), result.content, result.tokens


def parse_answer(raw_response: str) -> str:
    parsed = extract_json_object(raw_response)
    if parsed and parsed.get("answer") is not None:
        return str(parsed["answer"]).strip()
    return raw_response.strip()


def base_record(
    example: Example,
    *,
    method: str,
    config: dict[str, Any],
    build_tokens: int = 0,
    query_tokens: int = 0,
    build_time_seconds: float = 0.0,
    query_time_seconds: float = 0.0,
    hypothesis: str = "",
    raw_response: str | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "sample_id": example.sample_id,
        "memory_id": example.memory_id,
        "dataset": example.dataset,
        "question": example.question,
        "question_date": example.question_date,
        "question_type": example.question_type,
        "answer": example.answer,
        "hypothesis": hypothesis,
        "raw_response": raw_response,
        "method": method,
        "model": str(config["answer"]["name"]),
        "embedding_model": str((config.get("embedding") or {}).get("name") or ""),
        "build_tokens": int(build_tokens),
        "query_tokens": int(query_tokens),
        "build_time_seconds": round(float(build_time_seconds), 3),
        "query_time_seconds": round(float(query_time_seconds), 3),
        "error": error,
    }
    if extra:
        record.update(extra)
    return record


def already_done(path: Path, overwrite: bool) -> set[str]:
    if overwrite and path.exists():
        path.unlink()
    return set() if overwrite else completed_ids(path)


def write_record(path: Path, record: dict[str, Any]) -> None:
    append_jsonl(path, record)


def query_text(example: Example) -> str:
    if example.question_date:
        return f"Current Date: {example.question_date}\nQuestion: {example.question}"
    return example.question


def format_full_context(turns: list[Turn], *, max_chars: int = 0) -> str:
    blocks = []
    for index, turn in enumerate(turns, start=1):
        header = f"### Memory {index}"
        if turn.date:
            header += f"\nDate: {turn.date}"
        if turn.session_id:
            header += f"\nSession: {turn.session_id}"
        blocks.append(f"{header}\n{turn.role}: {turn.content}")
    return truncate_context("\n\n".join(blocks) or "None", max_chars=max_chars)


def format_retrieved_context(retrieved: list[RetrievedChunk], *, max_chars: int = 0) -> str:
    blocks = []
    for item in retrieved:
        header = f"### Memory {item.rank}"
        if item.date:
            header += f"\nDate: {item.date}"
        if item.session_id:
            header += f"\nSession: {item.session_id}"
        blocks.append(f"{header}\n{item.text}")
    return truncate_context("\n\n".join(blocks) or "None", max_chars=max_chars)


def truncate_context(text: str, *, max_chars: int = 0) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...[truncated]...\n"
    if max_chars <= len(marker) + 128:
        return text[:max_chars]
    budget = max_chars - len(marker)
    head = budget // 2
    tail = budget - head
    return f"{text[:head]}{marker}{text[-tail:]}"


def simple_answer_messages(example: Example, context: str) -> list[dict[str, str]]:
    prompt = f"""Answer the user's question using only the provided memory context.

User Question:
{query_text(example)}

Memory Context:
{context}

Rules:
1. Use only the memory context.
2. If the context is insufficient, say the provided information is not enough.
3. Keep the answer concise and specific.
4. Return only valid JSON.

Output JSON:
{{
  "reasoning": "one short sentence",
  "answer": "concise answer"
}}
"""
    return [{"role": "user", "content": prompt}]


def turn_chunks(turns: list[Turn]) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"turn-{index:05d}",
            text=f"{turn.role}: {turn.content}",
            date=turn.date,
            session_id=turn.session_id,
        )
        for index, turn in enumerate(turns)
    ]


def embedding_document(chunk: Chunk) -> str:
    if chunk.date:
        return f"Date: {chunk.date}\n{chunk.text}"
    return chunk.text


def dense_retrieve(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    top_k: int,
) -> list[RetrievedChunk]:
    if not chunks or embeddings.size == 0:
        return []
    scores = embeddings @ query_embedding.reshape(-1)
    order = np.argsort(scores)[::-1][:top_k]
    retrieved = []
    for rank, index in enumerate(order, start=1):
        chunk = chunks[int(index)]
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                date=chunk.date,
                session_id=chunk.session_id,
                score=float(scores[int(index)]),
                rank=rank,
            )
        )
    return retrieved


def group_by_memory(examples: Iterable[Example]) -> dict[str, list[Example]]:
    groups: dict[str, list[Example]] = {}
    for example in examples:
        groups.setdefault(example.memory_id, []).append(example)
    return groups


def now() -> float:
    return time.time()


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self.lock:
            write_record(self.path, record)

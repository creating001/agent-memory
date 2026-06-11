from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from agent_memory.core.schema import Chunk, Example
from external.baselines.common import (
    RuntimeClients,
    add_common_args,
    already_done,
    base_record,
    call_answer_model,
    dense_retrieve,
    embedding_document,
    format_retrieved_context,
    group_by_memory,
    load_filtered_examples,
    make_clients,
    now,
    query_text,
    simple_answer_messages,
    turn_chunks,
    write_record,
)


METHOD = "external_naive_rag_v1"


@dataclass(frozen=True)
class MemoryIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray
    build_tokens: int
    build_time_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the external naive-RAG baseline.")
    add_common_args(parser)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def build_memory_index(example: Example, clients: RuntimeClients) -> MemoryIndex:
    embedding_client = clients.embedding
    if embedding_client is None:
        raise ValueError("Naive RAG requires an embedding client.")

    started = now()
    chunks = turn_chunks(example.turns)
    embedded = embedding_client.embed_documents(embedding_document(chunk) for chunk in chunks)
    return MemoryIndex(
        chunks=chunks,
        embeddings=embedded.vectors,
        build_tokens=embedded.tokens,
        build_time_seconds=now() - started,
    )


def answer_example(
    example: Example,
    *,
    memory_index: MemoryIndex,
    clients: object,
    top_k: int,
    max_context_chars: int,
) -> dict:
    embedding_client = clients.embedding
    if embedding_client is None:
        raise ValueError("Naive RAG requires an embedding client.")

    started = now()
    query_embedding = embedding_client.embed_query(query_text(example))
    retrieved = dense_retrieve(
        memory_index.chunks,
        memory_index.embeddings,
        query_embedding.vectors[0],
        top_k=top_k,
    )
    context = format_retrieved_context(retrieved, max_chars=max_context_chars)
    hypothesis, raw_response, answer_tokens = call_answer_model(
        clients.answer,
        clients.config,
        simple_answer_messages(example, context),
    )
    return base_record(
        example,
        method=METHOD,
        config=clients.config,
        build_tokens=memory_index.build_tokens,
        query_tokens=query_embedding.tokens + answer_tokens,
        build_time_seconds=memory_index.build_time_seconds,
        query_time_seconds=now() - started,
        hypothesis=hypothesis,
        raw_response=raw_response,
        extra={
            "top_k": top_k,
            "num_chunks": len(memory_index.chunks),
            "answer_context_chunks": len(retrieved),
            "answer_context_chunk_ids": [item.chunk_id for item in retrieved],
            "retrieved_scores": [item.score for item in retrieved],
        },
    )


def main() -> None:
    args = parse_args()
    clients = make_clients(args.config, include_embedding=True)
    examples = load_filtered_examples(args)
    done = already_done(args.out, overwrite=args.overwrite)
    groups = group_by_memory(examples)

    total = len(examples)
    completed = 0
    for memory_id, group in groups.items():
        pending = [example for example in group if example.sample_id not in done]
        completed += len(group) - len(pending)
        if not pending:
            continue

        try:
            memory_index = build_memory_index(group[0], clients)
            print(f"[memory {memory_id}] built chunks={len(memory_index.chunks)}", flush=True)
        except Exception as exc:
            for example in pending:
                completed += 1
                write_record(
                    args.out,
                    base_record(example, method=METHOD, config=clients.config, error=repr(exc)),
                )
            if args.stop_on_error:
                raise
            continue

        for example in pending:
            completed += 1
            try:
                record = answer_example(
                    example,
                    memory_index=memory_index,
                    clients=clients,
                    top_k=args.top_k,
                    max_context_chars=args.max_context_chars,
                )
            except Exception as exc:
                record = base_record(example, method=METHOD, config=clients.config, error=repr(exc))
                if args.stop_on_error:
                    raise
            write_record(args.out, record)
            print(f"[{completed}/{total}] done {example.sample_id}", flush=True)


if __name__ == "__main__":
    main()

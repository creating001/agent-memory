from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agent_memory.baseline.chunking import build_chunks
from agent_memory.baseline.guardrails import apply_answer_guardrails, is_date_only_answer, normalize_relative_duration_record
from agent_memory.baseline.queries import (
    embedding_text,
    expanded_lexical_query_text,
    expanded_retrieval_query_text,
    prefer_user_chunks,
    rerank_retrieved,
    retrieval_query_text,
    select_retrieved_by_ranks,
    strategy_lexical_top_k,
)
from agent_memory.baseline.retrieve import merge_ranked_results, retrieve_lexical_top_k, retrieve_top_k
from agent_memory.baseline.routing import (
    METHOD_NAME,
    QuestionStrategy,
    RouteSettings,
    asks_explicit_assistant_memory,
    choose_lts_route,
    choose_strategy,
    effective_top_k,
    route_settings,
    should_use_strict_multi_aggregation,
)
from agent_memory.baseline.store import load_store, save_store
from agent_memory.core.embedding import EmbeddingClient
from agent_memory.core.llm import ChatClient
from agent_memory.core.schema import Example, RetrievedChunk
from agent_memory.prompts.answer import (
    answer_messages,
    evidence_table_answer_messages,
    evidence_table_messages,
    multi_evidence_answer_messages,
    multi_evidence_table_messages,
    parse_answer,
    verify_answer_messages,
)
from agent_memory.prompts.retrieve import evidence_rerank_messages, parse_reflective_queries, parse_selected_ranks
from agent_memory.prompts.retrieve import reflective_retrieval_messages


class NaiveRagBaseline:
    """LTS Agent Memory baseline.

    The pipeline has three stages:
    1. build raw turn memories and embeddings
    2. retrieve evidence with the LTS route settings
    3. answer, verify narrow order questions, then apply guardrails
    """

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        answer_client: ChatClient,
        config: dict[str, Any],
    ) -> None:
        self.embedding_client = embedding_client
        self.answer_client = answer_client
        self.config = config

    def build_memory(self, example: Example, store_dir: Path) -> dict[str, Any]:
        started = time.time()
        chunk_unit = str(self.config["retrieval"].get("chunk_unit", "turn"))
        chunks = build_chunks(example, chunk_unit=chunk_unit)
        embedded = self.embedding_client.embed_documents(embedding_text(chunk) for chunk in chunks)
        stats = {
            "memory_id": example.memory_id,
            "method": METHOD_NAME,
            "num_chunks": len(chunks),
            "chunk_unit": chunk_unit,
            "build_tokens": 0,
            "embedding_build_tokens": embedded.tokens,
            "build_time_seconds": round(time.time() - started, 3),
        }
        save_store(store_dir, chunks, embedded.vectors, stats)
        return stats

    def answer(self, example: Example, store_dir: Path) -> dict[str, Any]:
        route = choose_lts_route(example)
        record = self._answer_route(example, store_dir, route)

        if route == "order_reflective" and is_date_only_answer(str(record.get("hypothesis") or "")):
            route = "order_date_rewrite"
            record = self._answer_route(example, store_dir, "duration_temporal")

        record, route = normalize_relative_duration_record(record, route)
        return {
            **record,
            "route": route,
            "adaptive_route": "v44_single_branch",
        }

    def _answer_route(self, example: Example, store_dir: Path, route: str) -> dict[str, Any]:
        started = time.time()
        chunks, embeddings, build_stats = load_store(store_dir)

        settings = route_settings(route, self.config)
        strategy = choose_strategy(example.question, profile=settings.strategy_profile)
        top_k = effective_top_k(settings.top_k, strategy, settings.strategy_top_k)

        retrieved, retrieval_tokens, rerank_response = self.retrieve(
            example=example,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
            strategy=strategy,
            settings=settings,
        )
        answer_result = self.generate_answer(example, retrieved, strategy, settings)

        hypothesis = answer_result["hypothesis"]
        guarded_hypothesis = apply_answer_guardrails(example, retrieved, hypothesis, all_chunks=chunks)
        guardrail_response = None
        if guarded_hypothesis != hypothesis:
            guardrail_response = hypothesis
            hypothesis = guarded_hypothesis

        answer_cfg = self.config["answer"]
        return {
            "sample_id": example.sample_id,
            "memory_id": example.memory_id,
            "dataset": example.dataset,
            "question": example.question,
            "question_date": example.question_date,
            "question_type": example.question_type,
            "answer": example.answer,
            "hypothesis": hypothesis,
            "draft_hypothesis": answer_result["draft_hypothesis"],
            "raw_response": answer_result["raw_response"],
            "method": METHOD_NAME,
            "model": answer_cfg["name"],
            "embedding_model": self.config["embedding"]["name"],
            "chunk_unit": build_stats.get("chunk_unit"),
            "top_k": top_k,
            "num_chunks": build_stats.get("num_chunks", len(chunks)),
            "build_tokens": 0,
            "embedding_build_tokens": int(build_stats.get("embedding_build_tokens") or 0),
            "query_tokens": retrieval_tokens + int(answer_result["tokens"]),
            "build_time_seconds": float(build_stats.get("build_time_seconds", 0.0)),
            "query_time_seconds": round(time.time() - started, 3),
            "evidence_response": answer_result["evidence_response"],
            "profile_response": None,
            "rerank_response": rerank_response,
            "repair_response": None,
            "verify_response": answer_result["verify_response"],
            "guardrail_response": guardrail_response,
            "strategy": strategy.name,
            "error": None,
        }

    def retrieve(
        self,
        *,
        example: Example,
        chunks: list[Any],
        embeddings: Any,
        top_k: int,
        strategy: QuestionStrategy,
        settings: RouteSettings,
    ) -> tuple[list[RetrievedChunk], int, str | None]:
        retrieve_k = min(top_k * 2, 120) if settings.prefer_user_chunks else top_k
        retrieved = self.semantic_retrieve(chunks, embeddings, retrieval_query_text(example), retrieve_k)
        result_sets = [retrieved]
        tokens = 0
        rerank_response = None

        if strategy.use_expanded_query:
            result_sets.append(
                self.semantic_retrieve(chunks, embeddings, expanded_retrieval_query_text(example), retrieve_k)
            )

        if strategy.use_lexical:
            result_sets.append(self.lexical_retrieve(chunks, example, strategy))

        if len(result_sets) > 1:
            retrieved = merge_ranked_results(result_sets, retrieve_k)

        if settings.use_evidence_rerank and strategy.use_rerank:
            retrieved, rerank_response, rerank_tokens = self.rerank_evidence(example, retrieved, top_k)
            tokens += rerank_tokens

        if settings.use_reflective_retrieval and strategy.use_reflective:
            retrieved, reflection_tokens = self.reflective_retrieve(example, chunks, embeddings, retrieved, retrieve_k)
            tokens += reflection_tokens

        if settings.prefer_user_chunks:
            retrieved = prefer_user_chunks(example, strategy, retrieved, top_k)
        else:
            retrieved = rerank_retrieved(retrieved[:top_k])
        return retrieved, tokens, rerank_response

    def semantic_retrieve(
        self,
        chunks: list[Any],
        embeddings: Any,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedding_client.embed_query(query)
        return retrieve_top_k(chunks, embeddings, query_embedding.vectors[0], top_k)

    def lexical_retrieve(
        self,
        chunks: list[Any],
        example: Example,
        strategy: QuestionStrategy,
    ) -> list[RetrievedChunk]:
        lexical_top_k = strategy_lexical_top_k(
            strategy,
            int(self.config["retrieval"].get("hybrid_lexical_top_k", 5)),
        )
        return retrieve_lexical_top_k(
            chunks,
            expanded_lexical_query_text(example),
            lexical_top_k,
            include_date=bool(self.config["retrieval"].get("lexical_include_date", False)),
        )

    def rerank_evidence(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], str, int]:
        answer_cfg = self.config["answer"]
        keep = min(int(self.config["retrieval"].get("evidence_rerank_keep", 12)), top_k)
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=evidence_rerank_messages(example, retrieved, keep),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        selected = select_retrieved_by_ranks(retrieved, parse_selected_ranks(result.content), keep)
        return selected, result.content, result.tokens

    def reflective_retrieve(
        self,
        example: Example,
        chunks: list[Any],
        embeddings: Any,
        retrieved: list[RetrievedChunk],
        retrieve_k: int,
    ) -> tuple[list[RetrievedChunk], int]:
        answer_cfg = self.config["answer"]
        reflection = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=reflective_retrieval_messages(example, retrieved),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=final_max_tokens(answer_cfg),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )

        queries, keywords = parse_reflective_queries(reflection.content)
        result_sets = [retrieved]
        for query in queries:
            result_sets.append(self.semantic_retrieve(chunks, embeddings, query, retrieve_k))
        if keywords:
            result_sets.append(
                retrieve_lexical_top_k(
                    chunks,
                    " ".join(keywords),
                    retrieve_k,
                    include_date=bool(self.config["retrieval"].get("lexical_include_date", False)),
                )
            )
        if len(result_sets) == 1:
            return retrieved, reflection.tokens
        return merge_ranked_results(result_sets, retrieve_k), reflection.tokens

    def generate_answer(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        strategy: QuestionStrategy,
        settings: RouteSettings,
    ) -> dict[str, Any]:
        answer_cfg = self.config["answer"]
        max_context_chars = int(answer_cfg.get("max_context_chars", 12000))

        result, evidence_response, tokens = self.draft_answer(
            example=example,
            retrieved=retrieved,
            strategy=strategy,
            settings=settings,
            max_context_chars=max_context_chars,
        )
        hypothesis = parse_answer(result.content)
        draft_hypothesis = hypothesis

        verify_response = None
        if settings.use_verification and strategy.name in settings.verification_strategies:
            verify_result = self.answer_client.complete(
                model=str(answer_cfg["name"]),
                messages=verify_answer_messages(example, retrieved, hypothesis, max_context_chars=max_context_chars),
                temperature=float(answer_cfg["temperature"]),
                max_tokens=final_max_tokens(answer_cfg),
                thinking=str(answer_cfg.get("thinking", "default")),
                response_format={"type": "json_object"},
            )
            verify_response = verify_result.content
            tokens += verify_result.tokens
            hypothesis = parse_answer(verify_response)

        return {
            "hypothesis": hypothesis,
            "draft_hypothesis": draft_hypothesis,
            "raw_response": result.content,
            "evidence_response": evidence_response,
            "verify_response": verify_response,
            "tokens": tokens,
        }

    def draft_answer(
        self,
        *,
        example: Example,
        retrieved: list[RetrievedChunk],
        strategy: QuestionStrategy,
        settings: RouteSettings,
        max_context_chars: int,
    ) -> tuple[Any, str | None, int]:
        if strategy.name == "multi_evidence" and should_use_strict_multi_aggregation(example.question):
            return self.answer_with_multi_evidence_table(example, retrieved, max_context_chars)
        if settings.use_evidence_table and strategy.use_evidence_table:
            return self.answer_with_evidence_table(example, retrieved, max_context_chars)
        return self.answer_direct(example, retrieved, strategy, max_context_chars)

    def answer_with_multi_evidence_table(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        max_context_chars: int,
    ) -> tuple[Any, str, int]:
        answer_cfg = self.config["answer"]
        evidence_result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=multi_evidence_table_messages(example, retrieved, max_context_chars=max_context_chars),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=multi_evidence_answer_messages(example, evidence_result.content),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=final_max_tokens(answer_cfg),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        return result, evidence_result.content, evidence_result.tokens + result.tokens

    def answer_with_evidence_table(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        max_context_chars: int,
    ) -> tuple[Any, str, int]:
        answer_cfg = self.config["answer"]
        evidence_result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=evidence_table_messages(example, retrieved, max_context_chars=max_context_chars),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=evidence_table_answer_messages(example, evidence_result.content),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=final_max_tokens(answer_cfg),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        return result, evidence_result.content, evidence_result.tokens + result.tokens

    def answer_direct(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        strategy: QuestionStrategy,
        max_context_chars: int,
    ) -> tuple[Any, None, int]:
        answer_cfg = self.config["answer"]
        requirement_style = "assistant_recall" if asks_explicit_assistant_memory(example.question) else strategy.requirement_style
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=answer_messages(
                example,
                retrieved,
                requirement_style=requirement_style,
                max_context_chars=max_context_chars,
            ),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=final_max_tokens(answer_cfg),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        return result, None, result.tokens


def final_max_tokens(answer_cfg: dict[str, Any]) -> int:
    configured = int(answer_cfg.get("max_tokens", 8192))
    return int(answer_cfg.get("final_max_tokens", min(configured, 1024)))

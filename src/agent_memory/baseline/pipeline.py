from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent_memory.baseline.chunking import build_chunks
from agent_memory.baseline.context import (
    effective_context_window,
    effective_context_window_top_n,
    expand_retrieved_context,
    is_turn_pair_chunk_id,
    materialize_turn_pair_retrieval,
    maybe_prioritize_sum_financial_context,
    retrieval_search_space,
)
from agent_memory.baseline.guardrails import (
    apply_answer_guardrails,
    is_date_only_answer,
    normalize_relative_duration_record,
)
from agent_memory.baseline.evidence_finalizer import finalize_answer_from_evidence
from agent_memory.baseline.policy import (
    answer_guardrail_options,
    answer_max_context_chars,
    effective_multi_evidence_prompt_mode,
    final_max_tokens,
    map_route_plan,
    router_target_phrase_missing_candidate,
    should_force_direct_answer,
    should_force_evidence_table,
    should_force_strict_multi_evidence,
    should_include_question_date_in_query,
    should_include_router_target_support,
    should_use_answer_detail_requirements,
    should_use_context_relative_time_annotations,
    should_use_duration_evidence_prompt,
    should_use_factual_lexical,
    should_use_list_evidence_prompt,
    should_use_semantic_router_prompt_trace,
    temporal_query_hints,
    use_temporal_prompt,
)
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
    choose_memory_route,
    choose_strategy,
    effective_top_k,
    route_settings,
    should_use_strict_multi_aggregation,
    collection_intent_mode,
    router_mode,
    strategy_by_name,
    use_collection_intent_routing,
    preference_inference_mode,
    use_personalized_inference_routing,
)
from agent_memory.baseline.reranking import (
    apply_dedicated_rerank_anchor_retention,
    dedicated_rerank_pool_k,
    format_rerank_document,
    should_use_dedicated_rerank,
)
from agent_memory.baseline.store import load_store, save_store
from agent_memory.core.embedding import EmbeddingClient
from agent_memory.core.llm import ChatClient
from agent_memory.core.rerank import RerankClient
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
from agent_memory.prompts.router import parse_route_classification, route_classification_messages


class StrongMemoryBaseline:
    """Production-oriented Agent Memory baseline.

    The pipeline has three stages:
    1. build raw turn memories and embeddings
    2. route and retrieve evidence with the shared strong-memory policy
    3. compile evidence, answer, verify narrow cases, then apply guardrails
    """

    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        answer_client: ChatClient,
        rerank_client: RerankClient | None = None,
        config: dict[str, Any],
    ) -> None:
        self.embedding_client = embedding_client
        self.answer_client = answer_client
        self.rerank_client = rerank_client
        self.config = config

    def method_name(self) -> str:
        return str((self.config.get("retrieval") or {}).get("method") or METHOD_NAME)

    def build_memory(self, example: Example, store_dir: Path) -> dict[str, Any]:
        started = time.time()
        retrieval_cfg = self.config["retrieval"]
        chunk_unit = str(retrieval_cfg.get("chunk_unit", "turn"))
        relative_time_annotations = bool(retrieval_cfg.get("relative_time_annotations", False))
        turn_pair_chunks = bool(retrieval_cfg.get("turn_pair_chunks", False))
        turn_pair_max_chars = int(retrieval_cfg.get("turn_pair_max_chars", 2500))
        embedding_text_mode = str(retrieval_cfg.get("embedding_text_mode", "date_content"))
        chunks = build_chunks(
            example,
            chunk_unit=chunk_unit,
            relative_time_annotations=relative_time_annotations,
            turn_pair_chunks=turn_pair_chunks,
            turn_pair_max_chars=turn_pair_max_chars,
        )
        turn_pair_count = sum(1 for chunk in chunks if is_turn_pair_chunk_id(chunk.chunk_id))
        embedded = self.embedding_client.embed_documents(embedding_text(chunk, mode=embedding_text_mode) for chunk in chunks)
        stats = {
            "memory_id": example.memory_id,
            "method": self.method_name(),
            "num_chunks": len(chunks),
            "chunk_unit": chunk_unit,
            "embedding_text_mode": embedding_text_mode,
            "relative_time_annotations": relative_time_annotations,
            "turn_pair_chunks": turn_pair_chunks,
            "turn_pair_max_chars": turn_pair_max_chars,
            "num_turn_pair_chunks": turn_pair_count,
            "build_tokens": 0,
            "embedding_build_tokens": embedded.tokens,
            "build_time_seconds": round(time.time() - started, 3),
        }
        save_store(store_dir, chunks, embedded.vectors, stats)
        return stats

    def classify_route(self, example: Example) -> dict[str, Any] | None:
        retrieval_cfg = self.config.get("retrieval") or {}
        if str(retrieval_cfg.get("router_mode", "legacy")) not in {"llm_task_v1", "llm_task_selective_v1"}:
            return None

        answer_cfg = self.config["answer"]
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=route_classification_messages(
                example,
                include_temporal_hints=bool(retrieval_cfg.get("temporal_query_anchors", False)),
                include_target_support=should_include_router_target_support(answer_cfg, retrieval_cfg),
            ),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=min(final_max_tokens(answer_cfg), 512),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        parsed = parse_route_classification(result.content)
        parsed["_question"] = example.question
        route, strategy = map_route_plan(parsed, retrieval_cfg)
        return {
            **parsed,
            "route": route,
            "strategy": strategy,
            "tokens": result.tokens,
            "raw_response": result.content,
        }

    def should_apply_route_plan(self, route_plan: dict[str, Any] | None) -> bool:
        if not route_plan:
            return False
        retrieval_cfg = self.config.get("retrieval") or {}
        mode = str(retrieval_cfg.get("router_mode", "legacy"))
        if mode == "llm_task_v1":
            return True
        if mode != "llm_task_selective_v1":
            return False
        strategies = {str(value) for value in retrieval_cfg.get("semantic_router_apply_strategies", [])}
        tasks = {str(value) for value in retrieval_cfg.get("semantic_router_apply_tasks", [])}
        if strategies and str(route_plan.get("strategy")) in strategies:
            return True
        return bool(tasks and str(route_plan.get("task")) in tasks)

    def answer(self, example: Example, store_dir: Path) -> dict[str, Any]:
        route_plan = self.classify_route(example)
        route_plan_applied = self.should_apply_route_plan(route_plan)
        effective_route_plan = route_plan if route_plan_applied else None
        route_prompt_plan = route_plan if (
            route_plan_applied or should_use_semantic_router_prompt_trace(self.config)
        ) else None
        route = str(route_plan["route"]) if route_plan_applied and route_plan else choose_memory_route(example, self.config)
        record = self._answer_route(
            example,
            store_dir,
            route,
            route_plan=effective_route_plan,
            route_prompt_plan=route_prompt_plan,
            route_trace=route_plan,
            route_plan_applied=route_plan_applied,
        )

        if route == "order_reflective" and is_date_only_answer(str(record.get("hypothesis") or "")):
            route = "order_date_rewrite"
            record = self._answer_route(
                example,
                store_dir,
                "duration_temporal",
                route_plan=effective_route_plan,
                route_prompt_plan=route_prompt_plan,
                route_trace=route_plan,
                route_plan_applied=route_plan_applied,
            )

        record, route = normalize_relative_duration_record(record, route)
        return {
            **record,
            "route": route,
            "adaptive_route": "single_branch_memory",
        }

    def _answer_route(
        self,
        example: Example,
        store_dir: Path,
        route: str,
        route_plan: dict[str, Any] | None = None,
        route_prompt_plan: dict[str, Any] | None = None,
        route_trace: dict[str, Any] | None = None,
        route_plan_applied: bool = False,
    ) -> dict[str, Any]:
        started = time.time()
        chunks, embeddings, build_stats = load_store(store_dir)

        settings = route_settings(route, self.config)
        if route_plan and route_plan.get("strategy"):
            strategy = strategy_by_name(str(route_plan["strategy"]))
        else:
            strategy = choose_strategy(
                example.question,
                profile=settings.strategy_profile,
                collection_intent_routing=use_collection_intent_routing(self.config),
                personalized_inference_routing=use_personalized_inference_routing(self.config),
                preference_inference_mode=preference_inference_mode(self.config),
                routing_mode=router_mode(self.config),
                collection_intent_mode=collection_intent_mode(self.config),
            )
        top_k = effective_top_k(settings.top_k, strategy, settings.strategy_top_k)

        retrieved, retrieval_tokens, rerank_response, profile_response = self.retrieve(
            example=example,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
            strategy=strategy,
            settings=settings,
            route=route,
            route_trace=route_trace,
        )
        context_window = effective_context_window(settings, strategy, example.question, self.config)
        context_window_top_n = effective_context_window_top_n(settings, strategy, self.config)
        answer_context = expand_retrieved_context(
            chunks,
            retrieved,
            window=context_window,
            top_n=context_window_top_n,
        )
        answer_context = maybe_prioritize_sum_financial_context(self.config, example, answer_context, route_plan)
        answer_result = self.generate_answer(
            example,
            answer_context,
            strategy,
            settings,
            route_plan=route_prompt_plan,
            route_trace=route_trace,
        )
        answer_cfg = self.config["answer"]
        guardrail_options = answer_guardrail_options(answer_cfg)

        hypothesis = answer_result["hypothesis"]
        guarded_hypothesis = apply_answer_guardrails(
            example,
            answer_context,
            hypothesis,
            all_chunks=chunks,
            **guardrail_options,
            relative_temporal_evidence=answer_result["evidence_response"],
        )
        guardrail_response = None
        guardrail_candidate = None
        if guarded_hypothesis != hypothesis:
            guardrail_response = hypothesis
            guardrail_candidate = guarded_hypothesis
            hypothesis = guarded_hypothesis

        target_phrase_guard_response = None
        target_phrase_guard_candidate = None
        target_phrase_missing = router_target_phrase_missing_candidate(
            self.config,
            example,
            chunks,
            route_trace,
            strategy,
            hypothesis,
        )
        if target_phrase_missing:
            target_phrase_guard_response = hypothesis
            target_phrase_guard_candidate = target_phrase_missing
            hypothesis = target_phrase_missing

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
            "method": self.method_name(),
            "model": answer_cfg["name"],
            "embedding_model": self.config["embedding"]["name"],
            "chunk_unit": build_stats.get("chunk_unit"),
            "top_k": top_k,
            "context_window": context_window,
            "answer_temporal_prompt": bool(answer_result["temporal_prompt"]),
            **guardrail_options,
            "turn_pair_chunks": bool((self.config.get("retrieval") or {}).get("turn_pair_chunks", False)),
            "turn_pair_answer_mode": str(
                (self.config.get("retrieval") or {}).get("turn_pair_answer_mode", "direct")
            ),
            "sum_financial_context_prioritize": bool(answer_cfg.get("sum_financial_context_prioritize", False)),
            "temporal_query_anchor_mode": str(
                (self.config.get("retrieval") or {}).get("temporal_query_anchor_mode", "append")
            ),
            "question_date_query_mode": str(
                (self.config.get("retrieval") or {}).get("question_date_query_mode", "always")
            ),
            "question_date_query_included": should_include_question_date_in_query(self.config, strategy, route),
            "answer_context_chunks": len(answer_context),
            "answer_context_chunk_ids": [item.chunk_id for item in answer_context],
            "answer_context_session_ids": sorted({item.session_id for item in answer_context if item.session_id}),
            "num_chunks": build_stats.get("num_chunks", len(chunks)),
            "context_window_top_n": context_window_top_n,
            "build_tokens": int(build_stats.get("build_tokens") or 0),
            "embedding_build_tokens": int(build_stats.get("embedding_build_tokens") or 0),
            "query_tokens": int(route_trace.get("tokens", 0) if route_trace else 0)
            + retrieval_tokens
            + int(answer_result["tokens"]),
            "build_time_seconds": float(build_stats.get("build_time_seconds", 0.0)),
            "query_time_seconds": round(time.time() - started, 3),
            "evidence_response": answer_result["evidence_response"],
            "profile_response": profile_response,
            "rerank_response": rerank_response,
            "semantic_route_response": route_trace.get("raw_response") if route_trace else None,
            "semantic_route_task": route_trace.get("task") if route_trace else None,
            "semantic_route_operation": route_trace.get("operation") if route_trace else None,
            "semantic_route_answer_slot": route_trace.get("answer_slot") if route_trace else None,
            "semantic_route_prompt_trace": bool(route_prompt_plan and not route_plan_applied),
            "semantic_route_target_support_required": (
                route_trace.get("target_support_required") if route_trace else None
            ),
            "semantic_route_required_target_phrases": (
                route_trace.get("required_target_phrases") if route_trace else None
            ),
            "semantic_route_confidence": route_trace.get("confidence") if route_trace else None,
            "semantic_route_temporal_hints": route_trace.get("temporal_search_hints") if route_trace else None,
            "semantic_route_applied": route_plan_applied,
            "repair_response": None,
            "verify_response": answer_result["verify_response"],
            "target_verify_response": answer_result["target_verify_response"],
            "sum_verify_response": answer_result["sum_verify_response"],
            "structured_finalizer_response": answer_result["structured_finalizer_response"],
            "guardrail_response": guardrail_response,
            "guardrail_candidate": guardrail_candidate,
            "target_phrase_guard_response": target_phrase_guard_response,
            "target_phrase_guard_candidate": target_phrase_guard_candidate,
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
        route: str,
        route_trace: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedChunk], int, str | None, str | None]:
        retrieval_cfg = self.config.get("retrieval") or {}
        temporal_hints = temporal_query_hints(self.config, route_trace)
        anchor_mode = str(retrieval_cfg.get("temporal_query_anchor_mode", "append"))
        append_hints = temporal_hints if anchor_mode == "append" else []
        branch_hints = temporal_hints if anchor_mode == "branch" else []
        include_question_date = should_include_question_date_in_query(self.config, strategy, route)
        query_text = retrieval_query_text(
            example,
            temporal_hints=append_hints,
            include_question_date=include_question_date,
        )
        search_chunks, search_embeddings = retrieval_search_space(chunks, embeddings, strategy, retrieval_cfg)
        retrieve_k = min(top_k * 2, 120) if settings.prefer_user_chunks else top_k
        if self.rerank_client and should_use_dedicated_rerank(self.config, strategy, example, route_trace):
            retrieve_k = max(retrieve_k, dedicated_rerank_pool_k(retrieval_cfg, strategy))
        retrieved = self.semantic_retrieve(search_chunks, search_embeddings, query_text, retrieve_k)
        result_sets = [retrieved]
        tokens = 0
        rerank_response = None

        if branch_hints:
            result_sets.append(
                self.semantic_retrieve(
                    search_chunks,
                    search_embeddings,
                    retrieval_query_text(
                        example,
                        temporal_hints=branch_hints,
                        include_question_date=include_question_date,
                    ),
                    retrieve_k,
                )
            )

        if strategy.use_expanded_query:
            result_sets.append(
                self.semantic_retrieve(
                    search_chunks,
                    search_embeddings,
                    expanded_retrieval_query_text(
                        example,
                        temporal_hints=append_hints,
                        include_question_date=include_question_date,
                    ),
                    retrieve_k,
                )
            )

        if strategy.use_lexical or should_use_factual_lexical(self.config, strategy, route):
            result_sets.append(
                self.lexical_retrieve(
                    search_chunks,
                    example,
                    strategy,
                    temporal_hints=append_hints,
                    include_question_date=include_question_date,
                )
            )

        if len(result_sets) > 1:
            retrieved = merge_ranked_results(result_sets, retrieve_k)

        profile_response = None

        if settings.use_evidence_rerank and strategy.use_rerank:
            retrieved, rerank_response, rerank_tokens = self.rerank_evidence(example, retrieved, top_k)
            tokens += rerank_tokens

        if settings.use_reflective_retrieval and strategy.use_reflective:
            retrieved, reflection_tokens = self.reflective_retrieve(
                example,
                search_chunks,
                search_embeddings,
                retrieved,
                retrieve_k,
            )
            tokens += reflection_tokens

        if self.rerank_client and should_use_dedicated_rerank(self.config, strategy, example, route_trace):
            retrieved, dedicated_response = self.dedicated_rerank(
                example,
                retrieved,
                temporal_hints=append_hints,
                include_question_date=include_question_date,
            )
            rerank_response = dedicated_response

        retrieved = materialize_turn_pair_retrieval(chunks, retrieved, retrieval_cfg)

        if settings.prefer_user_chunks:
            retrieved = prefer_user_chunks(
                example,
                strategy,
                retrieved,
                top_k,
                non_user_keep=int(retrieval_cfg.get("prefer_user_non_user_keep", 0)),
            )
        else:
            retrieved = rerank_retrieved(retrieved[:top_k])
        return retrieved, tokens, rerank_response, profile_response

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
        *,
        temporal_hints: list[str] | None = None,
        include_question_date: bool = True,
    ) -> list[RetrievedChunk]:
        retrieval_cfg = self.config.get("retrieval") or {}
        lexical_top_k = strategy_lexical_top_k(
            strategy,
            int(retrieval_cfg.get("hybrid_lexical_top_k", 5)),
        )
        if strategy.name == "factual" and retrieval_cfg.get("factual_lexical_top_k") is not None:
            lexical_top_k = int(retrieval_cfg.get("factual_lexical_top_k"))
        if str(retrieval_cfg.get("lexical_expansion_mode", "legacy")) == "none":
            query = retrieval_query_text(
                example,
                temporal_hints=temporal_hints,
                include_question_date=include_question_date,
            )
        else:
            query = expanded_lexical_query_text(
                example,
                temporal_hints=temporal_hints,
                include_question_date=include_question_date,
                collection_query_expansion=bool(
                    retrieval_cfg.get("collection_query_expansion", False)
                ),
                scope_aware_collection_query_expansion=bool(
                    retrieval_cfg.get("collection_query_scope_aware_expansion", False)
                ),
            )
        return retrieve_lexical_top_k(
            chunks,
            query,
            lexical_top_k,
            include_date=bool(retrieval_cfg.get("lexical_include_date", False)),
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

    def dedicated_rerank(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        *,
        temporal_hints: list[str] | None = None,
        include_question_date: bool = True,
    ) -> tuple[list[RetrievedChunk], str]:
        if not self.rerank_client or not retrieved:
            return retrieved, ""
        query = retrieval_query_text(
            example,
            temporal_hints=temporal_hints,
            include_question_date=include_question_date,
        )
        rerank_cfg = self.config.get("rerank") or {}
        max_document_chars = int(rerank_cfg.get("document_max_chars", 0) or 0)
        result = self.rerank_client.rerank(
            query=query,
            documents=[
                format_rerank_document(item, max_chars=max_document_chars)
                for item in retrieved
            ],
        )
        scored = sorted(
            zip(retrieved, result.scores, strict=False),
            key=lambda pair: pair[1],
            reverse=True,
        )
        reranked = [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                text=item.text,
                date=item.date,
                session_id=item.session_id,
                score=float(score),
                rank=rank,
            )
            for rank, (item, score) in enumerate(scored, start=1)
        ]
        reranked = apply_dedicated_rerank_anchor_retention(
            original=retrieved,
            reranked=reranked,
            rerank_cfg=self.config.get("rerank") or {},
        )
        response = {
            "backend": "dedicated_rerank",
            "query": query,
            "include_question_date": include_question_date,
            "anchor_keep": int((self.config.get("rerank") or {}).get("anchor_keep", 0)),
            "anchor_after_top": int((self.config.get("rerank") or {}).get("anchor_after_top", 0)),
            "results": [
                {
                    "rank": rank,
                    "chunk_id": item.chunk_id,
                    "score": float(score),
                    "source_rank": item.rank,
                    "date": item.date,
                }
                for rank, (item, score) in enumerate(scored[:20], start=1)
            ],
            "usage": result.response.get("usage"),
        }
        return reranked, json.dumps(response, ensure_ascii=False)

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
        route_plan: dict[str, Any] | None = None,
        route_trace: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        answer_cfg = self.config["answer"]
        max_context_chars = answer_max_context_chars(answer_cfg, strategy)
        detailed_answer = should_use_answer_detail_requirements(
            answer_cfg,
            example,
            route_plan=route_plan,
            route_trace=route_trace,
        )

        result, evidence_response, tokens = self.draft_answer(
            example=example,
            retrieved=retrieved,
            strategy=strategy,
            settings=settings,
            max_context_chars=max_context_chars,
            route_plan=route_plan,
            detailed_answer=detailed_answer,
        )
        hypothesis = parse_answer(result.content)
        draft_hypothesis = hypothesis

        use_route_verification = settings.use_verification and strategy.name in settings.verification_strategies
        verification_max_words = int(answer_cfg.get("verification_max_draft_words", 0) or 0)
        if use_route_verification and verification_max_words > 0:
            use_route_verification = len(str(hypothesis or "").split()) <= verification_max_words

        verify_response = None
        if use_route_verification:
            verify_result = self.answer_client.complete(
                model=str(answer_cfg["name"]),
                messages=verify_answer_messages(
                    example,
                    retrieved,
                    hypothesis,
                    max_context_chars=max_context_chars,
                    context_relative_time_annotations=should_use_context_relative_time_annotations(
                        answer_cfg, strategy, example.question
                    ),
                ),
                temperature=float(answer_cfg["temperature"]),
                max_tokens=final_max_tokens(answer_cfg),
                thinking=str(answer_cfg.get("thinking", "default")),
                response_format={"type": "json_object"},
            )
            verify_response = verify_result.content
            tokens += verify_result.tokens
            hypothesis = parse_answer(verify_response)

        structured_finalizer_response = None
        if bool(answer_cfg.get("structured_evidence_finalizer", False)):
            finalization = finalize_answer_from_evidence(example, evidence_response, retrieved, hypothesis)
            if finalization:
                structured_finalizer_response = {
                    "reason": finalization.reason,
                    "before": hypothesis,
                    "after": finalization.answer,
                }
                hypothesis = finalization.answer

        target_verify_response = None
        sum_verify_response = None
        return {
            "hypothesis": hypothesis,
            "draft_hypothesis": draft_hypothesis,
            "raw_response": result.content,
            "evidence_response": evidence_response,
            "verify_response": verify_response,
            "target_verify_response": target_verify_response,
            "sum_verify_response": sum_verify_response,
            "structured_finalizer_response": structured_finalizer_response,
            "temporal_prompt": use_temporal_prompt(self.config, strategy, settings),
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
        route_plan: dict[str, Any] | None = None,
        detailed_answer: bool = False,
    ) -> tuple[Any, str | None, int]:
        answer_cfg = self.config["answer"]
        strict_multi_evidence = should_force_strict_multi_evidence(answer_cfg, example.question, strategy)
        if strategy.name == "multi_evidence" and (
            strict_multi_evidence
            or (
                bool(answer_cfg.get("force_strict_multi_aggregation", True))
                and should_use_strict_multi_aggregation(example.question)
            )
        ):
            return self.answer_with_multi_evidence_table(
                example,
                retrieved,
                strategy,
                max_context_chars,
                route_plan=route_plan,
                detailed_answer=detailed_answer,
            )
        if should_force_direct_answer(answer_cfg, strategy):
            return self.answer_direct(example, retrieved, strategy, max_context_chars, detailed_answer=detailed_answer)
        if settings.use_evidence_table and (
            strategy.use_evidence_table or should_force_evidence_table(answer_cfg, strategy)
        ):
            return self.answer_with_evidence_table(
                example,
                retrieved,
                strategy,
                settings,
                max_context_chars,
                route_plan=route_plan,
                detailed_answer=detailed_answer,
            )
        return self.answer_direct(example, retrieved, strategy, max_context_chars, detailed_answer=detailed_answer)

    def answer_with_multi_evidence_table(
        self,
        example: Example,
        retrieved: list[RetrievedChunk],
        strategy: QuestionStrategy,
        max_context_chars: int,
        route_plan: dict[str, Any] | None = None,
        detailed_answer: bool = False,
        extraction_max_tokens: int | None = None,
    ) -> tuple[Any, str, int]:
        answer_cfg = self.config["answer"]
        prompt_mode = effective_multi_evidence_prompt_mode(answer_cfg, example.question, route_plan=route_plan)
        evidence_result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=multi_evidence_table_messages(
                example,
                retrieved,
                max_context_chars=max_context_chars,
                mode=prompt_mode,
                context_relative_time_annotations=should_use_context_relative_time_annotations(
                    answer_cfg, strategy, example.question
                ),
                prompt_profile=str(answer_cfg.get("prompt_profile", "default")),
            ),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(extraction_max_tokens or answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=multi_evidence_answer_messages(
                example,
                evidence_result.content,
                mode=prompt_mode,
                detailed_answer=detailed_answer,
                prompt_profile=str(answer_cfg.get("prompt_profile", "default")),
            ),
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
        strategy: QuestionStrategy,
        settings: RouteSettings,
        max_context_chars: int,
        route_plan: dict[str, Any] | None = None,
        detailed_answer: bool = False,
        extraction_max_tokens: int | None = None,
    ) -> tuple[Any, str, int]:
        answer_cfg = self.config["answer"]
        temporal_prompt = use_temporal_prompt(self.config, strategy, settings)
        duration_prompt = bool(
            answer_cfg.get("duration_evidence_requirements", False)
        ) and should_use_duration_evidence_prompt(
            self.config,
            example.question,
            route_plan,
        )
        list_prompt = bool(answer_cfg.get("list_evidence_requirements", False)) and should_use_list_evidence_prompt(
            self.config,
            example.question,
            route_plan,
        )
        evidence_result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=evidence_table_messages(
                example,
                retrieved,
                max_context_chars=max_context_chars,
                temporal_reasoning=temporal_prompt,
                duration_reasoning=duration_prompt,
                list_reasoning=list_prompt,
                context_relative_time_annotations=should_use_context_relative_time_annotations(
                    answer_cfg, strategy, example.question
                ),
                prompt_profile=str(answer_cfg.get("prompt_profile", "default")),
            ),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=int(extraction_max_tokens or answer_cfg["max_tokens"]),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=evidence_table_answer_messages(
                example,
                evidence_result.content,
                temporal_reasoning=temporal_prompt,
                duration_reasoning=duration_prompt,
                list_reasoning=list_prompt,
                detailed_answer=detailed_answer,
                prompt_profile=str(answer_cfg.get("prompt_profile", "default")),
            ),
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
        *,
        detailed_answer: bool = False,
    ) -> tuple[Any, None, int]:
        answer_cfg = self.config["answer"]
        if asks_explicit_assistant_memory(example.question):
            requirement_style = "assistant_recall"
        else:
            requirement_style = strategy.requirement_style
        result = self.answer_client.complete(
            model=str(answer_cfg["name"]),
            messages=answer_messages(
                example,
                retrieved,
                requirement_style=requirement_style,
                max_context_chars=max_context_chars,
                context_relative_time_annotations=should_use_context_relative_time_annotations(
                    answer_cfg, strategy, example.question
                ),
                detailed_answer=detailed_answer,
                prompt_profile=str(answer_cfg.get("prompt_profile", "default")),
            ),
            temperature=float(answer_cfg["temperature"]),
            max_tokens=final_max_tokens(answer_cfg),
            thinking=str(answer_cfg.get("thinking", "default")),
            response_format={"type": "json_object"},
        )
        return result, None, result.tokens

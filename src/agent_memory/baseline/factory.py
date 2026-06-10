from __future__ import annotations

from typing import Any

from agent_memory.baseline.pipeline import StrongMemoryBaseline
from agent_memory.core.config import get_config
from agent_memory.core.embedding import EmbeddingClient
from agent_memory.core.llm import ChatClient
from agent_memory.core.rerank import RerankClient


def make_baseline(config: dict[str, Any], answer_api_key: str) -> StrongMemoryBaseline:
    """Build the production baseline from a resolved configuration dict."""

    embedding_cfg = config["embedding"]
    answer_cfg = config["answer"]
    rerank_client = make_rerank_client(config.get("rerank") or {})

    return StrongMemoryBaseline(
        embedding_client=EmbeddingClient(
            model=str(embedding_cfg["name"]),
            base_url=str(embedding_cfg["base_url"]),
            api_key=str(embedding_cfg.get("api_key", "EMPTY")),
            batch_size=int(embedding_cfg.get("batch_size", 64)),
            normalize=bool(embedding_cfg.get("normalize", True)),
            query_instruction=str(embedding_cfg.get("query_instruction", "")),
            max_input_bytes=int(embedding_cfg.get("max_input_bytes", 0)),
        ),
        answer_client=ChatClient(
            api_key=answer_api_key,
            base_url=str(get_config(config, "answer.base_url", "")),
            timeout_seconds=float(answer_cfg.get("timeout_seconds", 120)),
            max_retries=int(answer_cfg.get("max_retries", 2)),
            default_seed=optional_int(get_config(config, "answer.seed")),
            default_top_p=optional_float(get_config(config, "answer.top_p")),
        ),
        rerank_client=rerank_client,
        config=config,
    )


def make_rerank_client(rerank_cfg: dict[str, Any]) -> RerankClient | None:
    if not bool(rerank_cfg.get("enabled", False)):
        return None
    return RerankClient(
        model=str(rerank_cfg["name"]),
        base_url=str(rerank_cfg["base_url"]),
        api_key=str(rerank_cfg.get("api_key", "")),
        batch_size=int(rerank_cfg.get("batch_size", 0)),
        timeout_seconds=float(rerank_cfg.get("timeout_seconds", 120)),
    )


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

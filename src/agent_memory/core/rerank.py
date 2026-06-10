from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from agent_memory.core.llm import make_http_client


@dataclass(frozen=True)
class RerankResult:
    scores: list[float]
    response: dict[str, Any]
    tokens: int


class RerankClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "",
        batch_size: int = 0,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.batch_size = max(0, int(batch_size))
        self.client = make_http_client(base_url)
        self.timeout_seconds = timeout_seconds

    def rerank(self, *, query: str, documents: list[str]) -> RerankResult:
        if not documents:
            return RerankResult([], {"results": []}, 0)
        if self.batch_size <= 0 or self.batch_size >= len(documents):
            return self._rerank_batch(query=query, documents=documents, offset=0)

        scores = [float("-inf")] * len(documents)
        results: list[dict[str, Any]] = []
        tokens = 0
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            result = self._rerank_batch(query=query, documents=batch, offset=start)
            tokens += result.tokens
            for index, score in enumerate(result.scores, start=start):
                scores[index] = score
            results.extend(result.response.get("results") or [])
        return RerankResult(
            scores,
            {
                "id": None,
                "model": self.model,
                "usage": {"total_tokens": tokens},
                "results": results,
            },
            tokens,
        )

    def _rerank_batch(self, *, query: str, documents: list[str], offset: int) -> RerankResult:
        response = self.client.post(
            f"{self.base_url}/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
            json={"model": self.model, "query": query, "documents": documents},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        scores = [float("-inf")] * len(documents)
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if 0 <= index < len(scores):
                scores[index] = float(item.get("relevance_score", item.get("score", float("-inf"))))
        usage = payload.get("usage") if isinstance(payload, dict) else None
        tokens = int((usage or {}).get("total_tokens") or 0)
        return RerankResult(scores, compact_response(payload, index_offset=offset), tokens)


def compact_response(payload: dict[str, Any], *, index_offset: int = 0) -> dict[str, Any]:
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "usage": payload.get("usage"),
        "results": [
            {
                "index": int(item.get("index", 0)) + index_offset,
                "relevance_score": item.get("relevance_score", item.get("score")),
            }
            for item in payload.get("results") or []
            if isinstance(item, dict)
        ],
    }

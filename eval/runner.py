"""Eval runner: calls the backend search + agent APIs, returns scored results."""

from __future__ import annotations

import uuid
from typing import Literal

import httpx

from .scorer import RAGTriadScorer

SearchMode = Literal["semantic", "bm25", "hybrid", "hybrid+rerank"]


class EvalRunner:
    def __init__(
        self,
        scorer: RAGTriadScorer,
        base_url: str = "http://localhost:8000",
        search_limit: int = 5,
        timeout: float = 30.0,
    ):
        self._scorer = scorer
        self._base_url = base_url.rstrip("/")
        self._search_limit = search_limit
        self._timeout = timeout

    def _fetch_contexts(self, question: str, mode: str) -> list[dict]:
        url = f"{self._base_url}/search"
        resp = httpx.get(
            url,
            params={"q": question, "mode": mode, "limit": self._search_limit},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _fetch_answer(self, question: str) -> str:
        url = f"{self._base_url}/agent/chat"
        thread_id = str(uuid.uuid4())
        resp = httpx.post(
            url,
            json={"message": question, "thread_id": thread_id},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("reply", "")

    def run_question(self, question: str, mode: str) -> dict:
        chunks = self._fetch_contexts(question, mode)
        contexts = [c["text"] for c in chunks]
        answer = self._fetch_answer(question)
        scores = self._scorer.score_all(question, contexts, answer)
        return {
            "question": question,
            "mode": mode,
            "num_chunks": len(chunks),
            "answer": answer,
            "scores": scores,
        }

    def run_dataset(
        self,
        questions: list[dict],
        modes: list[str],
    ) -> list[dict]:
        results = []
        for q in questions:
            for mode in modes:
                try:
                    result = self.run_question(q["question"], mode)
                    result["id"] = q["id"]
                    result["difficulty"] = q.get("difficulty", "unknown")
                    results.append(result)
                except Exception as exc:
                    results.append(
                        {
                            "id": q["id"],
                            "question": q["question"],
                            "mode": mode,
                            "error": str(exc),
                            "scores": {
                                "context_relevance": 0.0,
                                "groundedness": 0.0,
                                "answer_relevance": 0.0,
                            },
                        }
                    )
        return results

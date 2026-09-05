"""RAG Triad scorer using an LLM-as-judge.

Metrics:
  context_relevance  - do the retrieved chunks contain info relevant to the question?
  groundedness       - is the answer supported by the retrieved contexts?
  answer_relevance   - does the answer actually address the question?

Each returns a float in [0.0, 1.0].
"""

from __future__ import annotations

_CONTEXT_RELEVANCE_PROMPT = """\
You are evaluating whether retrieved context is relevant to a question.

Question: {question}

Retrieved context chunks:
{contexts}

Rate how relevant the retrieved context is to the question on a scale of 1-5:
1 = completely irrelevant, no useful information
2 = mostly irrelevant, minor tangential mentions
3 = partially relevant, some useful information mixed with noise
4 = mostly relevant, most chunks are useful
5 = highly relevant, context directly addresses the question

Respond with only a single integer (1, 2, 3, 4, or 5). No explanation."""

_GROUNDEDNESS_PROMPT = """\
You are evaluating whether an answer is grounded in the provided context.

Context chunks:
{contexts}

Answer: {answer}

Rate how well the answer is supported by the context on a scale of 1-5:
1 = completely unsupported, contradicts context or fabricated
2 = mostly unsupported, claims go far beyond what context says
3 = partially supported, some claims backed by context
4 = mostly supported, minor unsupported details
5 = fully grounded, every claim directly supported by context

Respond with only a single integer (1, 2, 3, 4, or 5). No explanation."""

_ANSWER_RELEVANCE_PROMPT = """\
You are evaluating whether an answer addresses the question asked.

Question: {question}

Answer: {answer}

Rate how relevant the answer is to the question on a scale of 1-5:
1 = completely off-topic, does not address the question at all
2 = mostly off-topic, tangentially related
3 = partially relevant, addresses some aspects
4 = mostly relevant, addresses the main question with minor gaps
5 = fully relevant, directly and completely addresses the question

Respond with only a single integer (1, 2, 3, 4, or 5). No explanation."""


def _parse_score(response_text: str) -> float:
    text = response_text.strip()
    for ch in text:
        if ch.isdigit():
            val = int(ch)
            if 1 <= val <= 5:
                return (val - 1) / 4.0
    return 0.5


def _format_contexts(contexts: list[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))


class RAGTriadScorer:
    def __init__(self, llm_judge):
        """
        llm_judge: callable(prompt: str) -> str
            Takes a prompt string, returns text with a single integer 1-5.
        """
        self._judge = llm_judge

    def score_context_relevance(self, question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        prompt = _CONTEXT_RELEVANCE_PROMPT.format(
            question=question,
            contexts=_format_contexts(contexts),
        )
        return _parse_score(self._judge(prompt))

    def score_groundedness(self, answer: str, contexts: list[str]) -> float:
        if not contexts or not answer:
            return 0.0
        prompt = _GROUNDEDNESS_PROMPT.format(
            answer=answer,
            contexts=_format_contexts(contexts),
        )
        return _parse_score(self._judge(prompt))

    def score_answer_relevance(self, question: str, answer: str) -> float:
        if not answer:
            return 0.0
        prompt = _ANSWER_RELEVANCE_PROMPT.format(
            question=question,
            answer=answer,
        )
        return _parse_score(self._judge(prompt))

    def score_all(self, question: str, contexts: list[str], answer: str) -> dict:
        return {
            "context_relevance": self.score_context_relevance(question, contexts),
            "groundedness": self.score_groundedness(answer, contexts),
            "answer_relevance": self.score_answer_relevance(question, answer),
        }


def make_anthropic_judge(api_key: str, model: str = "claude-haiku-4-5-20251001"):
    """Return a judge callable backed by the Anthropic Messages API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def judge(prompt: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    return judge

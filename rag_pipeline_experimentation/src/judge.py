"""LLM-as-Judge for evaluating generated answers on 4 quality dimensions."""

from __future__ import annotations

from llm_utils.client import instructor_complete
from src.models import JudgeScore, QAResponse


def judge_answer(
    qa: QAResponse,
    model: str,
) -> JudgeScore:
    """Score a generated answer on 4 quality dimensions (1–5 scale).

    Args:
        qa:    The QAResponse to evaluate.
        model: Judge LLM model identifier.

    Returns:
        JudgeScore with per-dimension scores and brief reasoning.
    """
    chunk_summary = "\n".join(
        f"[{i+1}] {c.get('content', '')[:200]}"
        for i, c in enumerate(qa.chunks_used[:5])
    )

    prompt = (
        f"Question: {qa.query}\n\n"
        f"Context chunks:\n{chunk_summary}\n\n"
        f"Answer: {qa.answer}\n\n"
        "Score this answer on a 1–5 scale for each dimension:\n"
        "- relevance (1–5): Does the answer directly address the question?\n"
        "- accuracy (1–5): Is the information factually correct based on the context?\n"
        "- completeness (1–5): Does the answer fully cover the question?\n"
        "- citation_quality (1–5): Are sources cited correctly using [N] notation?\n"
        "Provide a brief reasoning string explaining your scores."
    )

    return instructor_complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert evaluator assessing RAG system answer quality. "
                    "Score strictly on evidence — do not reward hallucinated details. "
                    "Return scores as floats between 1.0 and 5.0."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_model=JudgeScore,
        model=model,
        temperature=0.0,
    )


def judge_batch(
    qa_pairs: list[QAResponse],
    model: str,
) -> list[JudgeScore]:
    """Judge a batch of QA pairs, returning a JudgeScore per pair."""
    scores: list[JudgeScore] = []
    for qa in qa_pairs:
        try:
            score = judge_answer(qa, model)
        except Exception as exc:
            print(f"  [judge] failed for query={qa.query[:40]!r}: {exc}")
            score = JudgeScore(
                relevance=0.0,
                accuracy=0.0,
                completeness=0.0,
                citation_quality=0.0,
                reasoning="evaluation failed",
            )
        scores.append(score)
    return scores

"""
Tests for visualizer.py.

No display is needed — tests only check that chart files are created and
that the DataFrame builder produces the expected columns/values.
"""

from __future__ import annotations

import pytest

from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModel,
    EvaluationResult, ExperimentConfig, MetricsResult, RetrievalConfig, RetrievalMethod,
)
from src.visualizer import _results_to_dataframe, generate_all_charts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
    embed_model: EmbedModel = EmbedModel.SMALL,
    retrieval_method: RetrievalMethod = RetrievalMethod.VECTOR,
    mrr: float = 0.8,
) -> EvaluationResult:
    cfg = ExperimentConfig(
        chunk=ChunkConfig(strategy=chunk_strategy),
        embed=EmbedConfig(model=embed_model),
        retrieval=RetrievalConfig(method=retrieval_method),
    )
    return EvaluationResult(
        experiment_id=cfg.experiment_id,
        config=cfg,
        metrics=MetricsResult(
            recall_at_k={"1": mrr * 0.8, "3": mrr * 0.9, "5": mrr, "10": mrr},
            precision_at_k={"1": mrr, "3": mrr / 3, "5": mrr / 5, "10": mrr / 10},
            mrr=mrr,
            map_score=mrr * 0.95,
            ndcg_at_k={"1": mrr * 0.85, "3": mrr * 0.9, "5": mrr * 0.95, "10": mrr},
            total_queries=20,
            avg_retrieval_time_s=0.05,
        ),
    )


def _mini_grid() -> list[EvaluationResult]:
    """Minimal 2×1×3 grid (6 results) — enough to exercise all charts."""
    results = []
    for chunk in [ChunkStrategy.FIXED_SIZE, ChunkStrategy.SENTENCE]:
        for retrieval, mrr in [
            (RetrievalMethod.VECTOR,  0.9),
            (RetrievalMethod.BM25,    0.7),
            (RetrievalMethod.HYBRID,  0.85),
        ]:
            results.append(_make_result(
                chunk_strategy=chunk,
                embed_model=EmbedModel.SMALL,
                retrieval_method=retrieval,
                mrr=mrr,
            ))
    return results


# ---------------------------------------------------------------------------
# _results_to_dataframe
# ---------------------------------------------------------------------------

class TestResultsToDataframe:
    def test_row_count_matches_results(self):
        results = _mini_grid()
        df = _results_to_dataframe(results)
        assert len(df) == len(results)

    def test_expected_columns_present(self):
        df = _results_to_dataframe(_mini_grid())
        for col in ["experiment_id", "chunk_label", "embed_label",
                    "retrieval_method", "mrr", "map_score",
                    "recall@1", "recall@5", "ndcg@5"]:
            assert col in df.columns, f"missing column: {col}"

    def test_mrr_values_correct(self):
        result = _make_result(mrr=0.75)
        df = _results_to_dataframe([result])
        assert abs(df.iloc[0]["mrr"] - 0.75) < 1e-6

    def test_recall_at_k_populated(self):
        df = _results_to_dataframe(_mini_grid())
        for k in [1, 3, 5, 10]:
            assert f"recall@{k}" in df.columns
            assert (df[f"recall@{k}"] >= 0).all()


# ---------------------------------------------------------------------------
# generate_all_charts
# ---------------------------------------------------------------------------

class TestGenerateAllCharts:
    def test_returns_nine_charts(self, tmp_path):
        saved = generate_all_charts(_mini_grid(), output_dir=tmp_path)
        assert len(saved) == 9

    def test_all_files_created(self, tmp_path):
        saved = generate_all_charts(_mini_grid(), output_dir=tmp_path)
        for name, path in saved.items():
            assert path.exists(), f"chart file missing: {name}"

    def test_files_are_png(self, tmp_path):
        saved = generate_all_charts(_mini_grid(), output_dir=tmp_path)
        for path in saved.values():
            assert path.suffix == ".png"

    def test_expected_chart_names(self, tmp_path):
        saved = generate_all_charts(_mini_grid(), output_dir=tmp_path)
        expected = {
            "mrr_leaderboard", "recall_at_k_curves", "chunking_comparison",
            "embedding_comparison", "retrieval_comparison", "mrr_heatmap",
            "recall_precision_scatter", "metric_correlation", "response_time_vs_quality",
        }
        assert set(saved.keys()) == expected

    def test_single_result_does_not_crash(self, tmp_path):
        saved = generate_all_charts([_make_result()], output_dir=tmp_path)
        assert len(saved) == 9

    def test_output_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "charts"
        generate_all_charts(_mini_grid(), output_dir=nested)
        assert nested.exists()

    def test_two_embed_models_in_comparison(self, tmp_path):
        results = [
            _make_result(embed_model=EmbedModel.SMALL, mrr=0.8),
            _make_result(embed_model=EmbedModel.LARGE, mrr=0.85),
        ]
        # Should not crash even with 1 retrieval method and 1 chunk strategy.
        saved = generate_all_charts(results, output_dir=tmp_path)
        assert (tmp_path / "embedding_comparison.png").exists()

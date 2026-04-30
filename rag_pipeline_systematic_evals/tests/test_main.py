"""
Tests for main.py CLI.

All I/O patched — no real PDF, API, or disk writes needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModel,
    EvaluationResult, ExperimentConfig, MetricsResult, RetrievalConfig, RetrievalMethod,
)
from src.main import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(mrr: float = 0.8) -> EvaluationResult:
    cfg = ExperimentConfig(
        chunk=ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE),
        embed=EmbedConfig(model=EmbedModel.SMALL),
        retrieval=RetrievalConfig(method=RetrievalMethod.VECTOR),
    )
    return EvaluationResult(
        experiment_id=cfg.experiment_id,
        config=cfg,
        metrics=MetricsResult(
            recall_at_k={"1": 0.7, "3": 0.8, "5": mrr, "10": mrr},
            precision_at_k={"1": 0.7, "3": 0.23, "5": mrr / 5, "10": mrr / 10},
            mrr=mrr, map_score=mrr * 0.95,
            ndcg_at_k={"1": 0.7, "3": 0.8, "5": mrr * 0.95, "10": mrr},
            total_queries=20,
        ),
    )


def _run_main(tmp_path: Path, extra_args: list[str] | None = None) -> int:
    fake_results = [_make_result(0.9), _make_result(0.7)]
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with (
        patch("src.main.run_grid_search", return_value=fake_results),
        patch("src.main.generate_all_charts", return_value={}),
    ):
        return main([str(fake_pdf)] + (extra_args or []))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_zero_on_success(self, tmp_path):
        assert _run_main(tmp_path) == 0

    def test_returns_one_for_missing_pdf(self, tmp_path):
        code = main([str(tmp_path / "nonexistent.pdf")])
        assert code == 1

    def test_no_charts_flag_skips_charts(self, tmp_path):
        fake_results = [_make_result()]
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF")

        with (
            patch("src.main.run_grid_search", return_value=fake_results),
            patch("src.main.generate_all_charts") as mock_charts,
        ):
            main([str(fake_pdf), "--no-charts"])

        mock_charts.assert_not_called()

    def test_force_flag_passed_to_grid_search(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF")

        with (
            patch("src.main.run_grid_search", return_value=[_make_result()]) as mock_gs,
            patch("src.main.generate_all_charts", return_value={}),
        ):
            main([str(fake_pdf), "--force", "--no-charts"])

        call_kwargs = mock_gs.call_args.kwargs
        assert call_kwargs.get("force") is True

    def test_n_pairs_passed_through(self, tmp_path):
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF")

        with (
            patch("src.main.run_grid_search", return_value=[_make_result()]) as mock_gs,
            patch("src.main.generate_all_charts", return_value={}),
        ):
            main([str(fake_pdf), "--n-pairs", "10", "--no-charts"])

        assert mock_gs.call_args.kwargs.get("n_pairs") == 10

    def test_top_n_limits_table_rows(self, tmp_path):
        # Smoke test — just check it doesn't crash with more results than top_n.
        fake_results = [_make_result(mrr=0.5 + i * 0.05) for i in range(8)]
        fake_pdf = tmp_path / "test.pdf"
        fake_pdf.write_bytes(b"%PDF")

        with (
            patch("src.main.run_grid_search", return_value=fake_results),
            patch("src.main.generate_all_charts", return_value={}),
        ):
            code = main([str(fake_pdf), "--top-n", "3"])
        assert code == 0

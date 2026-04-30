"""
Unit tests for rag_common.metrics.

Each test uses hand-calculable numbers so the expected values can be
verified without running the code.
"""

import math
import pytest
from rag_common import metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RETRIEVED = ["a", "b", "c", "d", "e"]  # ranked best → worst
RELEVANT_1 = {"a"}          # first result is relevant
RELEVANT_2 = {"b", "d"}     # 2nd and 4th are relevant
RELEVANT_NONE = {"z"}       # nothing retrieved is relevant
RELEVANT_ALL = {"a", "b", "c", "d", "e"}


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert metrics.recall_at_k(RETRIEVED, RELEVANT_1, k=5) == 1.0

    def test_partial_recall(self):
        # relevant = {b, d}, top-3 = {a,b,c} → hit b only → 1/2
        assert metrics.recall_at_k(RETRIEVED, RELEVANT_2, k=3) == pytest.approx(0.5)

    def test_zero_recall(self):
        assert metrics.recall_at_k(RETRIEVED, RELEVANT_NONE, k=5) == 0.0

    def test_empty_relevant(self):
        assert metrics.recall_at_k(RETRIEVED, set(), k=5) == 0.0

    def test_k_larger_than_retrieved(self):
        assert metrics.recall_at_k(["a"], RELEVANT_1, k=10) == 1.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision(self):
        # top-1 = [a], relevant = {a} → 1/1
        assert metrics.precision_at_k(RETRIEVED, RELEVANT_1, k=1) == 1.0

    def test_partial_precision(self):
        # top-5, relevant = {a} → 1/5
        assert metrics.precision_at_k(RETRIEVED, RELEVANT_1, k=5) == pytest.approx(0.2)

    def test_zero_precision(self):
        assert metrics.precision_at_k(RETRIEVED, RELEVANT_NONE, k=5) == 0.0

    def test_k_zero(self):
        assert metrics.precision_at_k(RETRIEVED, RELEVANT_1, k=0) == 0.0

    def test_two_relevant_in_top5(self):
        # relevant = {b, d}, top-5 → 2/5
        assert metrics.precision_at_k(RETRIEVED, RELEVANT_2, k=5) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:
    def test_first_position(self):
        assert metrics.reciprocal_rank(RETRIEVED, RELEVANT_1) == pytest.approx(1.0)

    def test_second_position(self):
        # relevant = {b}, b is at rank 2 → 1/2
        assert metrics.reciprocal_rank(RETRIEVED, {"b"}) == pytest.approx(0.5)

    def test_fifth_position(self):
        assert metrics.reciprocal_rank(RETRIEVED, {"e"}) == pytest.approx(0.2)

    def test_not_found(self):
        assert metrics.reciprocal_rank(RETRIEVED, RELEVANT_NONE) == 0.0

    def test_first_of_two_relevant(self):
        # {b, d} → first hit at rank 2 → 1/2
        assert metrics.reciprocal_rank(RETRIEVED, RELEVANT_2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# average_precision
# ---------------------------------------------------------------------------

class TestAveragePrecision:
    def test_single_relevant_first(self):
        # hit at rank 1, 1 relevant → AP = (1/1) / 1 = 1.0
        assert metrics.average_precision(RETRIEVED, RELEVANT_1) == pytest.approx(1.0)

    def test_single_relevant_second(self):
        # hit at rank 2 → AP = (1/2) / 1 = 0.5
        assert metrics.average_precision(RETRIEVED, {"b"}) == pytest.approx(0.5)

    def test_two_relevant(self):
        # relevant = {b, d}: hit at rank 2 (P=1/2), hit at rank 4 (P=2/4)
        # AP = (0.5 + 0.5) / 2 = 0.5
        assert metrics.average_precision(RETRIEVED, RELEVANT_2) == pytest.approx(0.5)

    def test_no_relevant_retrieved(self):
        assert metrics.average_precision(RETRIEVED, RELEVANT_NONE) == 0.0

    def test_empty_relevant(self):
        assert metrics.average_precision(RETRIEVED, set()) == 0.0

    def test_all_relevant(self):
        # perfect ranking → AP = mean of [1, 1, 1, 1, 1] = 1.0
        assert metrics.average_precision(RETRIEVED, RELEVANT_ALL) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_ndcg(self):
        # single relevant at rank 1 → DCG = IDCG = 1/log2(2) = 1.0
        assert metrics.ndcg_at_k(RETRIEVED, RELEVANT_1, k=5) == pytest.approx(1.0)

    def test_zero_ndcg(self):
        assert metrics.ndcg_at_k(RETRIEVED, RELEVANT_NONE, k=5) == 0.0

    def test_ndcg_two_relevant_perfect_order(self):
        # relevant = {a, b} at ranks 1,2 → perfect ranking → NDCG = 1.0
        assert metrics.ndcg_at_k(RETRIEVED, {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_ndcg_two_relevant_imperfect(self):
        # relevant = {b, d} at ranks 2, 4
        # DCG  = 1/log2(3) + 1/log2(5) ≈ 0.6309 + 0.4307 = 1.0616
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0    + 0.6309 = 1.6309
        # NDCG = 1.0616 / 1.6309 ≈ 0.6509
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        expected = dcg / idcg
        assert metrics.ndcg_at_k(RETRIEVED, RELEVANT_2, k=5) == pytest.approx(expected, rel=1e-6)

    def test_empty_relevant(self):
        assert metrics.ndcg_at_k(RETRIEVED, set(), k=5) == 0.0


# ---------------------------------------------------------------------------
# Aggregate: mrr, map_score
# ---------------------------------------------------------------------------

class TestMRR:
    def test_single_query_first(self):
        qr = [(RETRIEVED, RELEVANT_1)]
        assert metrics.mrr(qr) == pytest.approx(1.0)

    def test_two_queries(self):
        # query 1: first hit at rank 1 → RR = 1.0
        # query 2: first hit at rank 2 → RR = 0.5
        # MRR = (1.0 + 0.5) / 2 = 0.75
        qr = [(RETRIEVED, RELEVANT_1), (RETRIEVED, {"b"})]
        assert metrics.mrr(qr) == pytest.approx(0.75)

    def test_empty(self):
        assert metrics.mrr([]) == 0.0


class TestMAPScore:
    def test_perfect(self):
        qr = [(RETRIEVED, RELEVANT_1)]
        assert metrics.map_score(qr) == pytest.approx(1.0)

    def test_two_queries(self):
        # AP for {a}: 1.0; AP for {b}: 0.5 → MAP = 0.75
        qr = [(RETRIEVED, RELEVANT_1), (RETRIEVED, {"b"})]
        assert metrics.map_score(qr) == pytest.approx(0.75)

    def test_empty(self):
        assert metrics.map_score([]) == 0.0


# ---------------------------------------------------------------------------
# evaluate() convenience wrapper
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_keys_present(self):
        qr = [(RETRIEVED, RELEVANT_1)]
        result = metrics.evaluate(qr, k=5)
        assert set(result.keys()) == {"recall@5", "precision@5", "mrr", "map", "ndcg@5"}

    def test_perfect_run(self):
        qr = [(RETRIEVED, RELEVANT_1)]
        result = metrics.evaluate(qr, k=5)
        assert result["mrr"] == pytest.approx(1.0)
        assert result["recall@5"] == pytest.approx(1.0)
        assert result["ndcg@5"] == pytest.approx(1.0)

    def test_zero_run(self):
        qr = [(RETRIEVED, RELEVANT_NONE)]
        result = metrics.evaluate(qr, k=5)
        for v in result.values():
            assert v == pytest.approx(0.0)

    def test_custom_k(self):
        qr = [(RETRIEVED, RELEVANT_1)]
        result = metrics.evaluate(qr, k=3)
        assert "recall@3" in result
        assert "ndcg@3" in result

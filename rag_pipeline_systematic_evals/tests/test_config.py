import pytest
from src.config import (
    ChunkConfig, ChunkStrategy, EmbedConfig, EmbedModel,
    RetrievalConfig, RetrievalMethod, ExperimentConfig,
    build_experiment_grid, default_chunk_configs,
)


class TestChunkConfig:
    def test_label_fixed(self):
        c = ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256, overlap=50)
        assert c.label() == "fixed_256_ol50"

    def test_label_sentence(self):
        c = ChunkConfig(strategy=ChunkStrategy.SENTENCE, sentences_per_chunk=5, overlap_sentences=1)
        assert c.label() == "sentence_5s_ol1"

    def test_label_semantic(self):
        c = ChunkConfig(strategy=ChunkStrategy.SEMANTIC, breakpoint_threshold=0.65, max_sentences=10)
        assert "semantic" in c.label()

    def test_invalid_overlap_fixed(self):
        with pytest.raises(Exception):
            ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=100, overlap=100)

    def test_invalid_overlap_sentence(self):
        with pytest.raises(Exception):
            ChunkConfig(strategy=ChunkStrategy.SENTENCE, sentences_per_chunk=3, overlap_sentences=3)


class TestEmbedConfig:
    def test_label_small(self):
        assert EmbedConfig(model=EmbedModel.SMALL).label() == "small"

    def test_label_large(self):
        assert EmbedConfig(model=EmbedModel.LARGE).label() == "large"


class TestExperimentConfig:
    def test_experiment_id_format(self):
        cfg = ExperimentConfig(
            chunk=ChunkConfig(strategy=ChunkStrategy.FIXED_SIZE, chunk_size=256, overlap=50),
            embed=EmbedConfig(model=EmbedModel.SMALL),
            retrieval=RetrievalConfig(method=RetrievalMethod.VECTOR),
        )
        assert cfg.experiment_id == "fixed_256_ol50__small__vector"

    def test_experiment_id_unique_per_config(self):
        grid = build_experiment_grid()
        ids = [e.experiment_id for e in grid]
        assert len(ids) == len(set(ids)), "duplicate experiment IDs"


class TestBuildGrid:
    def test_default_grid_size(self):
        grid = build_experiment_grid()
        assert len(grid) == 24   # 4 × 2 × 3

    def test_all_combinations_present(self):
        grid = build_experiment_grid()
        methods = {e.retrieval.method for e in grid}
        assert methods == {RetrievalMethod.VECTOR, RetrievalMethod.BM25, RetrievalMethod.HYBRID}
        models = {e.embed.model for e in grid}
        assert models == {EmbedModel.SMALL, EmbedModel.LARGE}

    def test_custom_grid_size(self):
        grid = build_experiment_grid(
            chunk_configs=[default_chunk_configs()[0]],
            retrieval_configs=[RetrievalConfig(method=RetrievalMethod.VECTOR)],
        )
        assert len(grid) == 2   # 1 chunk × 2 embed × 1 retrieval

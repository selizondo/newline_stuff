"""
OpenAI batch embedder with disk cache for P3.

Design
------
Embedding 1000+ chunks via OpenAI is slow (~30 s) and costs money.
The cache stores embeddings keyed by (model, chunk_config_label) so
re-running grid_search with a different retrieval method skips re-embedding.

Cache layout on disk:
    data/embed_cache/{model_label}/{chunk_config_label}.pkl
    e.g. data/embed_cache/small/fixed_256_ol50.pkl

Each cache file is a dict:
    {"chunk_ids": [str, ...], "embeddings": np.ndarray (N, D)}

Chunk IDs are stored alongside vectors so the cache can be invalidated if
the chunk set changes (e.g. after a PDF re-parse with different cleaning).
"""

from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tiktoken
from openai import OpenAI

from rag_common.models import Chunk
from src.config import EmbedConfig

_MAX_TOKENS = 8191   # OpenAI embedding models hard limit

# Lazy-loaded — tiktoken.get_encoding() is slow (~200 ms); skip it at import time.
_tokenizer: tiktoken.Encoding | None = None


def _get_tokenizer() -> tiktoken.Encoding:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


def _truncate(text: str) -> str:
    enc = _get_tokenizer()
    tokens = enc.encode(text)
    if len(tokens) <= _MAX_TOKENS:
        return text
    return enc.decode(tokens[:_MAX_TOKENS])


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Core embedding functions
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: list[Chunk],
    config: EmbedConfig,
    chunk_config_label: str,
    force_reembed: bool = False,
) -> np.ndarray:
    """
    Embed `chunks` using the OpenAI model specified in `config`.

    Returns an ndarray of shape (N, D) in the same order as `chunks`.
    Reads from cache when available; writes cache after embedding.

    Args:
        chunks:             list of Chunk objects to embed.
        config:             EmbedConfig specifying model, batch_size, cache_dir.
        chunk_config_label: label from ChunkConfig.label() used as the cache key.
                            Must be unique per chunking configuration to avoid
                            serving stale embeddings when chunk boundaries change.
        force_reembed:      skip cache and re-embed even if cached data exists.
    """
    chunk_ids = [c.id_str() for c in chunks]
    cache_path = _cache_path(config, chunk_config_label)

    if not force_reembed and cache_path.exists():
        cached = _load_cache(cache_path)
        if cached["chunk_ids"] == chunk_ids:
            return cached["embeddings"]

    embeddings = _batch_embed([_truncate(c.content) for c in chunks], config)
    _save_cache(cache_path, chunk_ids, embeddings)
    return embeddings


def embed_texts(texts: list[str], config: EmbedConfig) -> np.ndarray:
    """Embed arbitrary texts using the model in `config`. Not cached."""
    return _batch_embed([_truncate(t) for t in texts], config)


def embed_query(query: str, model: str) -> np.ndarray:
    """Embed a single query string. Not cached (queries are ephemeral)."""
    client = _get_client()
    response = client.embeddings.create(input=[_truncate(query)], model=model)
    return np.array(response.data[0].embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def _batch_embed(texts: list[str], config: EmbedConfig) -> np.ndarray:
    """
    Send texts to OpenAI in batches; reassemble into a single matrix.

    ThreadPoolExecutor parallelises batches so total wall time is roughly
    (n_batches / n_workers) × per_batch_latency instead of n_batches × latency.
    OpenAI's embedding endpoint is safe for concurrent requests.
    """
    batches = [
        texts[i : i + config.batch_size]
        for i in range(0, len(texts), config.batch_size)
    ]

    results: list[np.ndarray] = [None] * len(batches)   # type: ignore[list-item]

    def _embed_batch(args: tuple[int, list[str]]) -> None:
        idx, batch = args
        client = _get_client()
        for attempt in range(2):
            try:
                response = client.embeddings.create(input=batch, model=config.model.value)
                results[idx] = np.array(
                    [d.embedding for d in response.data], dtype=np.float32
                )
                return
            except Exception as exc:
                if attempt == 0:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Embedding batch {idx} failed: {exc}") from exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_embed_batch, enumerate(batches)))

    return np.vstack(results)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(config: EmbedConfig, chunk_label: str) -> Path:
    return config.cache_dir / config.label() / f"{chunk_label}.pkl"


def _save_cache(path: Path, chunk_ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "embeddings": embeddings}, f)


def _load_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def cache_exists(config: EmbedConfig, chunk_config_label: str) -> bool:
    return _cache_path(config, chunk_config_label).exists()

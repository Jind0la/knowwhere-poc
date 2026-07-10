"""knowwhere.providers.embeddings — Pluggable embedding providers.

Three backends, one interface:
    OllamaEmbeddingProvider  — local via Ollama (nomic-embed-text)
    OpenAIEmbeddingProvider  — cloud via OpenAI (text-embedding-3-small)
    LocalEmbeddingProvider   — local via sentence-transformers (all-MiniLM-L6-v2)

Auto-detection from config:
    from knowwhere.providers.embeddings import get_embedding_provider
    provider = get_embedding_provider({"embedding": {"provider": "local"}})
    vec = provider.embed("hello world")  # → np.ndarray(384,) or None
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# nomic-embed-text outputs 768d; we Matryoshka-truncate to 256d.
# all-MiniLM-L6-v2 outputs 384d — no truncation needed.
# text-embedding-3-small outputs up to 1536d — truncate to configured dim.
DEFAULT_TRUNC_DIM = 256
OLLAMA_DIM = 768
LOCAL_DIM = 384
OPENAI_DIM = 1536


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec)) or 1.0
    return (vec / norm).astype(np.float32)


class EmbeddingProvider(Protocol):
    """Protocol for embedding backends."""

    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> np.ndarray | None:
        """Return L2-normalized embedding vector or None on failure."""
        ...


# ═══════════════════════════════════════════════════════════════════
# Ollama (local, nomic-embed-text)
# ═══════════════════════════════════════════════════════════════════

class OllamaEmbeddingProvider:
    """Embed via local Ollama instance (nomic-embed-text, 768d → 256d)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        trunc_dim: int = DEFAULT_TRUNC_DIM,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.trunc_dim = trunc_dim
        self.timeout = timeout

    @property
    def dimension(self) -> int:
        return self.trunc_dim

    def embed(self, text: str) -> np.ndarray | None:
        if not (text or "").strip():
            return None
        payload = json.dumps({
            "model": self.model,
            "input": [text[:2000]],
        }).encode()
        url = f"{self.base_url}/api/embed"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            emb = np.array(data["embeddings"][0], dtype=np.float32)
            if len(emb) > self.trunc_dim:
                emb = emb[:self.trunc_dim]
            return _normalize(emb)
        except (urllib.error.URLError, TimeoutError, KeyError,
                json.JSONDecodeError) as exc:
            logger.warning("Ollama embed failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════════════
# OpenAI / compatible API
# ═══════════════════════════════════════════════════════════════════

class OpenAIEmbeddingProvider:
    """Embed via OpenAI-compatible API (text-embedding-3-small, 1536d → 256d).

    Also works with DeepSeek, Grok, or any /v1/embeddings endpoint.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        trunc_dim: int = DEFAULT_TRUNC_DIM,
        timeout: int = 30,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.trunc_dim = trunc_dim
        self.timeout = timeout

    @property
    def dimension(self) -> int:
        return self.trunc_dim

    def embed(self, text: str) -> np.ndarray | None:
        if not self.api_key:
            logger.warning("OpenAI embedding skipped: no API key")
            return None
        if not (text or "").strip():
            return None
        payload = json.dumps({
            "input": [text[:8000]],
            "model": self.model,
        }).encode()
        url = f"{self.base_url}/embeddings"
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
            emb = np.array(data["data"][0]["embedding"], dtype=np.float32)
            if len(emb) > self.trunc_dim:
                emb = emb[:self.trunc_dim]
            return _normalize(emb)
        except Exception as exc:
            logger.warning("OpenAI embed failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════════════
# Local (sentence-transformers, offline)
# ═══════════════════════════════════════════════════════════════════

class LocalEmbeddingProvider:
    """Embed via local sentence-transformers model (all-MiniLM-L6-v2, 384d).

    Zero external calls. Requires: pip install knowwhere[embeddings]
    Downloads ~80 MB model on first use.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        trunc_dim: int | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model: object | None = None
        # all-MiniLM-L6-v2 is 384d — no truncation by default
        self._trunc_dim = trunc_dim

    @property
    def dimension(self) -> int:
        return self._trunc_dim or LOCAL_DIM

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
            logger.info("Loaded local embedding model: %s", self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install knowwhere[embeddings]"
            )

    def embed(self, text: str) -> np.ndarray | None:
        if not (text or "").strip():
            return None
        try:
            self._load_model()
            emb = self._model.encode(  # type: ignore[union-attr]
                [text[:2000]],
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0].astype(np.float32)
            if self._trunc_dim and len(emb) > self._trunc_dim:
                emb = emb[:self._trunc_dim]
            return emb
        except Exception as exc:
            logger.warning("Local embed failed: %s", exc)
            return None


# ═══════════════════════════════════════════════════════════════════
# Auto-detection from config
# ═══════════════════════════════════════════════════════════════════

def get_embedding_provider(
    config: dict | None = None,
    *,
    provider: str | None = None,
) -> EmbeddingProvider:
    """Auto-detect and instantiate the right embedding provider.

    Priority: explicit `provider` arg > config['embedding']['provider'] > env var > Ollama

    Each provider uses its own sensible defaults:
    - Ollama: nomic-embed-text 768d → Matryoshka 256d
    - OpenAI: text-embedding-3-small 1536d → 256d
    - Local: all-MiniLM-L6-v2 native 384d (no truncation)

    For custom dimensions, construct the provider directly:
        OllamaEmbeddingProvider(trunc_dim=768)

    Args:
        config: Full KnowWhere config dict (from config.toml).
        provider: Override provider name ('ollama', 'openai', 'local').

    Returns:
        An EmbeddingProvider instance.
    """
    if not provider and config:
        provider = config.get("embedding", {}).get("provider", "")

    if not provider:
        # Auto-detect: try Ollama, fall back to local
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2):
                provider = "ollama"
        except Exception:
            try:
                import sentence_transformers  # noqa: F401
                provider = "local"
            except ImportError:
                provider = "ollama"  # best-effort default

    if provider == "openai":
        return OpenAIEmbeddingProvider()  # 1536d → 256d by default
    elif provider == "local":
        return LocalEmbeddingProvider()   # native 384d, no truncation
    else:
        return OllamaEmbeddingProvider()  # 768d → 256d Matryoshka

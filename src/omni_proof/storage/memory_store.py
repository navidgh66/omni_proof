"""In-memory vector store for testing and offline development."""

import numpy as np

from omni_proof.storage.vector_store import VectorStore


class InMemoryVectorStore(VectorStore):
    """VectorStore backed by in-memory dicts with numpy cosine similarity search."""

    def __init__(self, default_namespace: str = "creatives"):
        self._default_namespace = default_namespace
        # namespace -> {asset_id: {"embedding": list[float], "metadata": dict}}
        self._data: dict[str, dict[str, dict]] = {}

    def _ns(self, namespace: str | None) -> str:
        return namespace or self._default_namespace

    async def upsert(
        self, asset_id: str, embedding: list[float], metadata: dict, namespace: str | None = None
    ) -> None:
        ns = self._ns(namespace)
        if ns not in self._data:
            self._data[ns] = {}
        self._data[ns][asset_id] = {"embedding": embedding, "metadata": metadata}

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        namespace: str | None = None,
    ) -> list[dict]:
        ns = self._ns(namespace)
        entries = self._data.get(ns, {})
        if not entries:
            return []

        query = np.array(query_embedding)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        scored = []
        for asset_id, entry in entries.items():
            # Apply filters (simple $eq matching)
            if filters and not self._matches_filters(entry["metadata"], filters):
                continue
            emb = np.array(entry["embedding"])
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                continue
            score = float(np.dot(query, emb) / (query_norm * emb_norm))
            scored.append({"id": asset_id, "score": score, "metadata": entry["metadata"]})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _matches_filters(metadata: dict, filters: dict) -> bool:
        for key, condition in filters.items():
            if isinstance(condition, dict) and "$eq" in condition:
                if metadata.get(key) != condition["$eq"]:
                    return False
            elif metadata.get(key) != condition:
                return False
        return True

    async def delete(self, asset_id: str, namespace: str | None = None) -> None:
        ns = self._ns(namespace)
        if ns in self._data:
            self._data[ns].pop(asset_id, None)

    async def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None:
        ns = self._ns(namespace)
        if ns not in self._data:
            self._data[ns] = {}
        for asset_id, embedding, metadata in vectors:
            self._data[ns][asset_id] = {"embedding": embedding, "metadata": metadata}

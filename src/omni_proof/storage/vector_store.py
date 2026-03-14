"""Vector store interface and Pinecone implementation."""

import asyncio
from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    async def upsert(
        self,
        asset_id: str,
        embedding: list[float],
        metadata: dict,
        namespace: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        namespace: str | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    async def delete(self, asset_id: str, namespace: str | None = None) -> None: ...

    @abstractmethod
    async def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None: ...


class PineconeVectorStore(VectorStore):
    """Pinecone serverless vector store implementation."""

    def __init__(self, index, default_namespace: str = "creatives"):
        self._index = index
        self._default_namespace = default_namespace

    async def upsert(
        self,
        asset_id: str,
        embedding: list[float],
        metadata: dict,
        namespace: str | None = None,
    ) -> None:
        ns = namespace or self._default_namespace
        await asyncio.to_thread(
            self._index.upsert,
            vectors=[(asset_id, embedding, metadata)],
            namespace=ns,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filters: dict | None = None,
        namespace: str | None = None,
    ) -> list[dict]:
        ns = namespace or self._default_namespace
        kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": ns,
        }
        if filters:
            kwargs["filter"] = filters
        results = await asyncio.to_thread(self._index.query, **kwargs)
        return [
            {"id": match.id, "score": match.score, "metadata": match.metadata}
            for match in results.matches
        ]

    async def delete(self, asset_id: str, namespace: str | None = None) -> None:
        ns = namespace or self._default_namespace
        await asyncio.to_thread(self._index.delete, ids=[asset_id], namespace=ns)

    async def upsert_batch(
        self,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str | None = None,
        batch_size: int = 100,
    ) -> None:
        ns = namespace or self._default_namespace
        await asyncio.to_thread(
            self._index.upsert,
            vectors=vectors,
            namespace=ns,
            batch_size=batch_size,
        )

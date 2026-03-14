import pytest

from omni_proof.storage.memory_store import InMemoryVectorStore


class TestInMemoryVectorStore:
    @pytest.fixture
    def store(self):
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_upsert_and_search(self, store):
        await store.upsert("a1", [1.0, 0.0, 0.0], {"type": "ad"})
        await store.upsert("a2", [0.0, 1.0, 0.0], {"type": "ad"})
        results = await store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "a1"
        assert results[0]["score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_search_with_filter(self, store):
        await store.upsert("a1", [1.0, 0.0], {"type": "guideline"})
        await store.upsert("a2", [1.0, 0.0], {"type": "creative"})
        results = await store.search([1.0, 0.0], filters={"type": {"$eq": "guideline"}})
        assert len(results) == 1
        assert results[0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.upsert("a1", [1.0, 0.0], {})
        await store.delete("a1")
        results = await store.search([1.0, 0.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store):
        vectors = [("a1", [1.0, 0.0], {"k": "v"}), ("a2", [0.0, 1.0], {"k": "v2"})]
        await store.upsert_batch(vectors)
        results = await store.search([1.0, 0.0], top_k=10)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, store):
        await store.upsert("a1", [1.0], {}, namespace="ns1")
        await store.upsert("a2", [1.0], {}, namespace="ns2")
        results = await store.search([1.0], namespace="ns1")
        assert len(results) == 1
        assert results[0]["id"] == "a1"

    @pytest.mark.asyncio
    async def test_empty_search(self, store):
        results = await store.search([1.0, 0.0])
        assert results == []

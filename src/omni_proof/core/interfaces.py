"""Abstract base classes for embedding providers and other core interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path


class EmbeddingProvider(ABC):
    """Abstract interface for generating embeddings from content."""

    @abstractmethod
    async def generate_embedding(self, content: str | Path, dimensions: int = 3072) -> list[float]:
        """Generate embedding vector for given content.

        Args:
            content: Text string or path to media file (image, video, audio, PDF)
            dimensions: Output dimensionality (e.g., 128, 768, 1536, 3072 for Matryoshka)

        Returns:
            Embedding vector as list of floats

        Raises:
            ValueError: If dimensions not supported by provider
        """
        pass

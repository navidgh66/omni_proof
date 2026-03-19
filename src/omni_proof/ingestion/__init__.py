"""Ingestion layer: preprocessing, embedding, and pipeline."""

from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.ingestion.pipeline import IngestPipeline
from omni_proof.ingestion.preprocessor import AssetPreprocessor

__all__ = ["AssetPreprocessor", "GeminiClient", "IngestPipeline"]

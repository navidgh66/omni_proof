"""OmniProof: Causal-Multimodal Engine for Creative Performance Attribution."""

from importlib.metadata import version

__version__ = version("omni-proof")

from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import BrandProfile
from omni_proof.causal.base import Estimator
from omni_proof.causal.estimator import DMLEstimator
from omni_proof.config.settings import Settings
from omni_proof.core.interfaces import EmbeddingProvider
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.orchestration.compliance_chain import ComplianceChain
from omni_proof.orchestration.insight_synthesizer import InsightSynthesizer
from omni_proof.storage.memory_store import InMemoryVectorStore
from omni_proof.storage.vector_store import PineconeVectorStore, VectorStore

__all__ = [
    "BrandExtractor",
    "BrandProfile",
    "ComplianceChain",
    "DMLEstimator",
    "EmbeddingProvider",
    "Estimator",
    "GeminiClient",
    "InMemoryVectorStore",
    "InsightSynthesizer",
    "PineconeVectorStore",
    "Settings",
    "VectorStore",
]

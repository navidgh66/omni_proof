"""Custom exception hierarchy for OmniProof."""


class OmniProofError(Exception):
    """Base exception for all OmniProof errors."""


class IngestionError(OmniProofError):
    """Base exception for ingestion-related errors."""


class EmbeddingError(IngestionError):
    """Exception raised when embedding generation fails."""


class MetadataExtractionError(IngestionError):
    """Exception raised when metadata extraction fails."""


class StorageError(OmniProofError):
    """Base exception for storage-related errors."""


class VectorStoreError(StorageError):
    """Exception raised for vector store operations."""


class RelationalStoreError(StorageError):
    """Exception raised for relational store operations."""


class CausalError(OmniProofError):
    """Base exception for causal inference errors."""


class DAGConstructionError(CausalError):
    """Exception raised when DAG construction fails."""


class EstimationError(CausalError):
    """Exception raised when causal estimation fails."""


class RefutationError(CausalError):
    """Exception raised when refutation fails."""


class ComplianceError(OmniProofError):
    """Exception raised for compliance-related errors."""

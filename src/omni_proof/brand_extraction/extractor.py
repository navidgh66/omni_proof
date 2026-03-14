"""BrandExtractor orchestrator."""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import structlog

from omni_proof.brand_extraction.asset_processor import AssetProcessor
from omni_proof.brand_extraction.conflict_detector import ConflictDetector
from omni_proof.brand_extraction.models import BrandConflict, BrandProfile
from omni_proof.brand_extraction.pattern_aggregator import PatternAggregator
from omni_proof.core.interfaces import EmbeddingProvider
from omni_proof.ingestion.gemini_client import GeminiClient
from omni_proof.storage.vector_store import VectorStore

logger = structlog.get_logger()


class BrandExtractor:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        gemini_client: GeminiClient,
        vector_store: VectorStore,
    ):
        self._embedding = embedding_provider
        self._vector_store = vector_store
        self._processor = AssetProcessor(embedding_provider, gemini_client)
        self._aggregator = PatternAggregator()
        self._conflict_detector = ConflictDetector()

    async def extract(self, brand_name: str, assets: list[Path]) -> BrandProfile:
        extractions = await self._processor.process_batch(assets)
        rules, voice, visual_style, fingerprint, confidence = self._aggregator.aggregate(
            extractions
        )

        now = datetime.now(timezone.utc)
        profile = BrandProfile(
            profile_id=str(uuid4()),
            brand_name=brand_name,
            rules=rules,
            voice=voice,
            visual_style=visual_style,
            visual_fingerprint=fingerprint,
            source_assets=[str(p) for p in assets],
            extractions=extractions,
            confidence_scores=confidence,
            created_at=now,
            updated_at=now,
        )

        await self._index_profile(profile)
        logger.info("brand_extracted", brand=brand_name, rules=len(rules), assets=len(assets))
        return profile

    async def update(
        self,
        profile: BrandProfile,
        new_assets: list[Path],
    ) -> tuple[BrandProfile, list[BrandConflict]]:
        new_extractions = await self._processor.process_batch(new_assets)
        all_extractions = list(profile.extractions) + new_extractions

        rules, voice, visual_style, fingerprint, confidence = self._aggregator.aggregate(
            all_extractions
        )
        conflicts = self._conflict_detector.detect(profile, rules, voice, visual_style)

        now = datetime.now(timezone.utc)
        updated = BrandProfile(
            profile_id=profile.profile_id,
            brand_name=profile.brand_name,
            rules=rules,
            voice=voice,
            visual_style=visual_style,
            visual_fingerprint=fingerprint,
            source_assets=profile.source_assets + [str(p) for p in new_assets],
            extractions=all_extractions,
            confidence_scores=confidence,
            created_at=profile.created_at,
            updated_at=now,
        )

        logger.info(
            "brand_updated",
            brand=profile.brand_name,
            new_assets=len(new_assets),
            conflicts=len(conflicts),
        )
        return updated, conflicts

    async def _index_profile(self, profile: BrandProfile) -> None:
        await self._vector_store.upsert(
            asset_id=f"fingerprint:{profile.profile_id}",
            embedding=profile.visual_fingerprint,
            metadata={
                "source_type": "brand_fingerprint",
                "brand_name": profile.brand_name,
                "profile_id": profile.profile_id,
            },
            namespace="brand_assets",
        )
        for rule in profile.rules:
            rule_embedding = await self._embedding.generate_embedding(
                rule.description, task_type="RETRIEVAL_DOCUMENT"
            )
            await self._vector_store.upsert(
                asset_id=rule.rule_id,
                embedding=rule_embedding,
                metadata={
                    "source_type": "brand_rule",
                    "section_type": rule.section_type,
                    "brand_name": profile.brand_name,
                    "profile_id": profile.profile_id,
                },
                namespace="brand_assets",
            )

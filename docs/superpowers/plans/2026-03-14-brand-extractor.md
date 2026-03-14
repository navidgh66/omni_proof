# BrandExtractor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multimodal brand identity extraction pipeline that creates structured BrandProfiles from collections of assets (PDFs, images, videos, audio), with incremental update and conflict detection.

**Architecture:** Three-stage pipeline (asset processing -> pattern aggregation -> profile assembly) with two user-triggered operations: fresh extraction and incremental update. Uses Gemini Flash for structured metadata extraction and Gemini Embedding 2 for multimodal embeddings in a unified semantic space.

**Tech Stack:** Python 3.11+, Pydantic models, numpy for embedding math, existing GeminiClient/EmbeddingProvider/VectorStore interfaces.

**Spec:** `docs/superpowers/specs/2026-03-14-brand-extractor-design.md`

---

## Task 0: Prerequisites — EmbeddingProvider task_type parameter

**Files:**
- Modify: `src/omni_proof/core/interfaces.py`
- Modify: `src/omni_proof/ingestion/gemini_client.py`
- Modify: `tests/unit/test_gemini_client.py`

This adds the `task_type` parameter to the embedding interface so brand extraction can request task-optimized embeddings.

- [ ] **Step 1: Update EmbeddingProvider ABC**

In `src/omni_proof/core/interfaces.py`, add `task_type: str | None = None` parameter:

```python
from abc import ABC, abstractmethod
from pathlib import Path


class EmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(
        self,
        content: str | Path,
        dimensions: int = 3072,
        task_type: str | None = None,
    ) -> list[float]: ...
```

- [ ] **Step 2: Update GeminiClient to accept and pass task_type**

In `src/omni_proof/ingestion/gemini_client.py`, update `generate_embedding`:

```python
async def generate_embedding(
    self,
    content: str | Path,
    dimensions: int = DEFAULT_EMBEDDING_DIMS,
    task_type: str | None = None,
) -> list[float]:
    if dimensions not in MATRYOSHKA_DIMS:
        raise ValueError(f"dimensions must be one of {MATRYOSHKA_DIMS}, got {dimensions}")

    config = {"output_dimensionality": dimensions}
    if task_type:
        config["task_type"] = task_type

    for attempt in range(self._max_retries):
        try:
            response = await self._client.aio.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=str(content),
                config=config,
            )
            return response.embeddings[0].values
        except Exception as e:
            wait = 2**attempt
            logger.warning("embed_retry", attempt=attempt, wait=wait, error=str(e))
            if attempt == self._max_retries - 1:
                raise EmbeddingError(
                    f"Embedding failed after {self._max_retries} retries"
                ) from e
            await asyncio.sleep(wait)
```

- [ ] **Step 3: Add test for task_type parameter**

In `tests/unit/test_gemini_client.py`, add:

```python
@pytest.mark.asyncio
async def test_task_type_passed_to_config(self, client):
    mock_response = MagicMock()
    mock_response.embeddings = [MagicMock(values=[0.1] * 3072)]
    client._client = MagicMock()
    client._client.aio.models.embed_content = AsyncMock(return_value=mock_response)

    await client.generate_embedding(Path("test.jpg"), task_type="SEMANTIC_SIMILARITY")

    call_args = client._client.aio.models.embed_content.call_args
    assert call_args.kwargs["config"]["task_type"] == "SEMANTIC_SIMILARITY"
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_gemini_client.py -v --tb=short`
Expected: All tests pass including the new one.

- [ ] **Step 5: Commit**

```bash
git add src/omni_proof/core/interfaces.py src/omni_proof/ingestion/gemini_client.py tests/unit/test_gemini_client.py
git commit -m "feat: add task_type parameter to EmbeddingProvider and GeminiClient"
```

---

## Task 1: Data Models

**Files:**
- Create: `src/omni_proof/brand_extraction/__init__.py`
- Create: `src/omni_proof/brand_extraction/models.py`
- Test: `tests/unit/test_brand_extraction_models.py`

All Pydantic models for the brand extraction pipeline. No logic, just data structures.

- [ ] **Step 1: Create module init**

Create `src/omni_proof/brand_extraction/__init__.py` (empty file).

- [ ] **Step 2: Write model tests**

Create `tests/unit/test_brand_extraction_models.py`:

```python
"""Tests for brand extraction data models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandConflict,
    BrandProfile,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule


def _make_metadata() -> BrandAssetMetadata:
    return BrandAssetMetadata(
        asset_description="A social media ad with blue branding",
        colors=BrandColorInfo(hex_codes=["#004E89", "#FFFFFF"], palette_mood="cool"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"],
            font_names=["Helvetica"],
            text_hierarchy="strong_hierarchy",
        ),
        tone=BrandToneInfo(
            formality="formal",
            emotional_register="authoritative",
            key_phrases=["Trust the process"],
            vocabulary_themes=["reliability", "innovation"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered",
            motion_intensity="static",
            dominant_objects=["logo", "product"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )


class TestBrandAssetMetadata:
    def test_valid_construction(self):
        meta = _make_metadata()
        assert meta.colors.hex_codes == ["#004E89", "#FFFFFF"]
        assert meta.logo_detected is True

    def test_json_roundtrip(self):
        meta = _make_metadata()
        data = meta.model_dump()
        restored = BrandAssetMetadata(**data)
        assert restored == meta


class TestAssetExtraction:
    def test_valid_construction(self):
        ext = AssetExtraction(
            asset_path="/tmp/ad.jpg",
            media_type="image",
            embedding=[0.1] * 3072,
            structured_metadata=_make_metadata(),
            extracted_at=datetime.now(timezone.utc),
        )
        assert ext.media_type == "image"
        assert len(ext.embedding) == 3072


class TestBrandProfile:
    def test_valid_construction(self):
        now = datetime.now(timezone.utc)
        profile = BrandProfile(
            profile_id="bp-1",
            brand_name="TestBrand",
            rules=[
                BrandRule(
                    rule_id="r1",
                    section_type="color_palette",
                    description="Use blue and white",
                    hex_codes=["#004E89", "#FFFFFF"],
                ),
            ],
            voice=BrandVoice(
                formality="formal",
                emotional_register="authoritative",
                vocabulary_themes=["reliability"],
                sentence_style="short_punchy",
                confidence=0.85,
            ),
            visual_style=BrandVisualStyle(
                dominant_colors=["#004E89", "#FFFFFF"],
                color_consistency=0.9,
                typography_styles=["sans-serif"],
                layout_patterns=["centered"],
                motion_style="static",
                confidence=0.8,
            ),
            visual_fingerprint=[0.1] * 3072,
            source_assets=["/tmp/ad1.jpg", "/tmp/ad2.jpg"],
            extractions=[],
            confidence_scores={"color": 0.9, "typography": 0.8},
            created_at=now,
            updated_at=now,
        )
        assert profile.brand_name == "TestBrand"
        assert len(profile.rules) == 1


class TestBrandConflict:
    def test_valid_construction(self):
        conflict = BrandConflict(
            dimension="color_palette",
            existing_value="#004E89",
            new_value="#FF6B35",
            source_assets=["/tmp/new_ad.jpg"],
            severity="major",
        )
        assert conflict.severity == "major"
```

- [ ] **Step 3: Run tests to see them fail**

Run: `.venv/bin/pytest tests/unit/test_brand_extraction_models.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'omni_proof.brand_extraction'`

- [ ] **Step 4: Create models.py**

Create `src/omni_proof/brand_extraction/models.py`:

```python
"""Data models for brand identity extraction pipeline."""

from datetime import datetime

from pydantic import BaseModel, Field

from omni_proof.rag.models import BrandRule  # noqa: F401 — re-export for convenience


class BrandColorInfo(BaseModel):
    hex_codes: list[str]
    palette_mood: str

class BrandTypographyInfo(BaseModel):
    font_styles: list[str]
    font_names: list[str]
    text_hierarchy: str

class BrandToneInfo(BaseModel):
    formality: str
    emotional_register: str
    key_phrases: list[str]
    vocabulary_themes: list[str]

class BrandVisualInfo(BaseModel):
    layout_pattern: str
    motion_intensity: str
    dominant_objects: list[str]

class BrandAssetMetadata(BaseModel):
    asset_description: str
    colors: BrandColorInfo
    typography: BrandTypographyInfo
    tone: BrandToneInfo
    visual: BrandVisualInfo
    logo_detected: bool
    media_type_detected: str

class AssetExtraction(BaseModel):
    asset_path: str
    media_type: str
    embedding: list[float]
    structured_metadata: BrandAssetMetadata
    extracted_at: datetime

class BrandVoice(BaseModel):
    formality: str
    emotional_register: str
    vocabulary_themes: list[str]
    sentence_style: str
    confidence: float

class BrandVisualStyle(BaseModel):
    dominant_colors: list[str]
    color_consistency: float
    typography_styles: list[str]
    layout_patterns: list[str]
    motion_style: str
    confidence: float

class BrandConflict(BaseModel):
    dimension: str
    existing_value: str
    new_value: str
    source_assets: list[str]
    severity: str

class BrandProfile(BaseModel):
    profile_id: str
    brand_name: str
    rules: list[BrandRule]
    voice: BrandVoice
    visual_style: BrandVisualStyle
    visual_fingerprint: list[float]
    source_assets: list[str]
    extractions: list[AssetExtraction] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest tests/unit/test_brand_extraction_models.py -v --tb=short`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/omni_proof/brand_extraction/ tests/unit/test_brand_extraction_models.py
git commit -m "feat: add brand extraction data models"
```

---

## Task 2: PatternAggregator

**Files:**
- Create: `src/omni_proof/brand_extraction/pattern_aggregator.py`
- Test: `tests/unit/test_pattern_aggregator.py`

Pure logic component. No external calls, no mocking needed. This is the core intelligence of the pipeline.

- [ ] **Step 1: Write tests**

Create `tests/unit/test_pattern_aggregator.py`:

```python
"""Tests for brand pattern aggregation."""

from collections import Counter
from datetime import datetime, timezone

import numpy as np
import pytest

from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
)
from omni_proof.brand_extraction.pattern_aggregator import PatternAggregator


def _make_extraction(
    path: str = "/tmp/ad.jpg",
    hex_codes: list[str] | None = None,
    font_styles: list[str] | None = None,
    formality: str = "formal",
    emotional_register: str = "authoritative",
    layout: str = "centered",
    embedding: list[float] | None = None,
) -> AssetExtraction:
    return AssetExtraction(
        asset_path=path,
        media_type="image",
        embedding=embedding or [0.1] * 10,
        structured_metadata=BrandAssetMetadata(
            asset_description="Test asset",
            colors=BrandColorInfo(
                hex_codes=hex_codes or ["#004E89", "#FFFFFF"],
                palette_mood="cool",
            ),
            typography=BrandTypographyInfo(
                font_styles=font_styles or ["sans-serif"],
                font_names=["Helvetica"],
                text_hierarchy="strong_hierarchy",
            ),
            tone=BrandToneInfo(
                formality=formality,
                emotional_register=emotional_register,
                key_phrases=["Trust us"],
                vocabulary_themes=["reliability"],
            ),
            visual=BrandVisualInfo(
                layout_pattern=layout,
                motion_intensity="static",
                dominant_objects=["logo"],
            ),
            logo_detected=True,
            media_type_detected="image",
        ),
        extracted_at=datetime.now(timezone.utc),
    )


class TestPatternAggregator:
    def setup_method(self):
        self.aggregator = PatternAggregator()

    def test_empty_extractions_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self.aggregator.aggregate([])

    def test_single_asset_extraction(self):
        ext = _make_extraction(hex_codes=["#004E89", "#FFFFFF"])
        rules, voice, visual, fingerprint, confidence = self.aggregator.aggregate([ext])

        assert any(r.section_type == "color_palette" for r in rules)
        color_rule = next(r for r in rules if r.section_type == "color_palette")
        assert "#004E89" in color_rule.hex_codes

        assert voice.formality == "formal"
        assert voice.confidence == 1.0

        assert "#004E89" in visual.dominant_colors
        assert len(fingerprint) == 10

    def test_color_frequency_aggregation(self):
        exts = [
            _make_extraction(hex_codes=["#004E89", "#FFFFFF"]),
            _make_extraction(hex_codes=["#004E89", "#FF6B35"]),
            _make_extraction(hex_codes=["#004E89", "#000000"]),
        ]
        rules, _, visual, _, confidence = self.aggregator.aggregate(exts)

        # #004E89 appears in all 3 assets — should be dominant
        assert visual.dominant_colors[0] == "#004E89"
        assert confidence["color"] > 0.5

    def test_voice_consistency(self):
        exts = [
            _make_extraction(formality="formal", emotional_register="authoritative"),
            _make_extraction(formality="formal", emotional_register="authoritative"),
            _make_extraction(formality="casual", emotional_register="playful"),
        ]
        _, voice, _, _, confidence = self.aggregator.aggregate(exts)

        # Majority is formal/authoritative
        assert voice.formality == "formal"
        assert voice.emotional_register == "authoritative"
        assert confidence["voice"] == pytest.approx(2 / 3, abs=0.01)

    def test_visual_fingerprint_is_normalized(self):
        emb1 = list(np.random.randn(10))
        emb2 = list(np.random.randn(10))
        exts = [
            _make_extraction(embedding=emb1),
            _make_extraction(embedding=emb2),
        ]
        _, _, _, fingerprint, _ = self.aggregator.aggregate(exts)

        norm = np.linalg.norm(fingerprint)
        assert norm == pytest.approx(1.0, abs=0.01)

    def test_typography_aggregation(self):
        exts = [
            _make_extraction(font_styles=["sans-serif"]),
            _make_extraction(font_styles=["sans-serif", "serif"]),
            _make_extraction(font_styles=["sans-serif"]),
        ]
        _, _, visual, _, _ = self.aggregator.aggregate(exts)

        assert "sans-serif" in visual.typography_styles
```

- [ ] **Step 2: Run tests to see them fail**

Run: `.venv/bin/pytest tests/unit/test_pattern_aggregator.py -v --tb=short`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement PatternAggregator**

Create `src/omni_proof/brand_extraction/pattern_aggregator.py`:

```python
"""Aggregates per-asset extractions into brand-level patterns."""

from collections import Counter
from uuid import uuid4

import numpy as np
import structlog

from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule

logger = structlog.get_logger()


class PatternAggregator:
    """Aggregates structured metadata and embeddings across assets into brand patterns."""

    def aggregate(
        self,
        extractions: list[AssetExtraction],
    ) -> tuple[list[BrandRule], BrandVoice, BrandVisualStyle, list[float], dict[str, float]]:
        if not extractions:
            raise ValueError("Cannot aggregate empty extractions list")

        rules = []
        confidence_scores = {}

        # Color aggregation
        color_counter: Counter[str] = Counter()
        for ext in extractions:
            for hex_code in ext.structured_metadata.colors.hex_codes:
                color_counter[hex_code] += 1
        dominant_colors = [c for c, _ in color_counter.most_common(5)]
        color_confidence = (
            sum(1 for ext in extractions if any(c in ext.structured_metadata.colors.hex_codes for c in dominant_colors[:3]))
            / len(extractions)
        )
        confidence_scores["color"] = color_confidence
        if dominant_colors:
            rules.append(
                BrandRule(
                    rule_id=str(uuid4()),
                    section_type="color_palette",
                    description=f"Brand colors: {', '.join(dominant_colors)}",
                    hex_codes=dominant_colors,
                )
            )

        # Typography aggregation
        style_counter: Counter[str] = Counter()
        font_counter: Counter[str] = Counter()
        for ext in extractions:
            for s in ext.structured_metadata.typography.font_styles:
                style_counter[s] += 1
            for f in ext.structured_metadata.typography.font_names:
                font_counter[f] += 1
        dominant_styles = [s for s, _ in style_counter.most_common(3)]
        dominant_fonts = [f for f, _ in font_counter.most_common(5)]
        typo_confidence = (
            sum(1 for ext in extractions if any(s in ext.structured_metadata.typography.font_styles for s in dominant_styles[:1]))
            / len(extractions)
        ) if dominant_styles else 0.0
        confidence_scores["typography"] = typo_confidence
        if dominant_fonts or dominant_styles:
            rules.append(
                BrandRule(
                    rule_id=str(uuid4()),
                    section_type="typography",
                    description=f"Typography: {', '.join(dominant_styles)}. Fonts: {', '.join(dominant_fonts)}",
                    approved_fonts=dominant_fonts,
                )
            )

        # Voice aggregation
        formality_counter: Counter[str] = Counter()
        register_counter: Counter[str] = Counter()
        all_themes: list[str] = []
        for ext in extractions:
            formality_counter[ext.structured_metadata.tone.formality] += 1
            register_counter[ext.structured_metadata.tone.emotional_register] += 1
            all_themes.extend(ext.structured_metadata.tone.vocabulary_themes)

        dominant_formality = formality_counter.most_common(1)[0][0]
        dominant_register = register_counter.most_common(1)[0][0]
        unique_themes = list(dict.fromkeys(all_themes))
        voice_agreement = (
            sum(
                1
                for ext in extractions
                if ext.structured_metadata.tone.formality == dominant_formality
                and ext.structured_metadata.tone.emotional_register == dominant_register
            )
            / len(extractions)
        )
        confidence_scores["voice"] = voice_agreement

        # Determine sentence style from formality
        sentence_style = "short_punchy" if dominant_formality == "casual" else "mixed"

        voice = BrandVoice(
            formality=dominant_formality,
            emotional_register=dominant_register,
            vocabulary_themes=unique_themes[:20],
            sentence_style=sentence_style,
            confidence=voice_agreement,
        )

        if unique_themes:
            rules.append(
                BrandRule(
                    rule_id=str(uuid4()),
                    section_type="tone",
                    description=f"Brand voice: {dominant_formality}, {dominant_register}. Themes: {', '.join(unique_themes[:5])}",
                    tone_keywords=unique_themes[:10],
                )
            )

        # Visual style aggregation
        layout_counter: Counter[str] = Counter()
        motion_counter: Counter[str] = Counter()
        for ext in extractions:
            layout_counter[ext.structured_metadata.visual.layout_pattern] += 1
            motion_counter[ext.structured_metadata.visual.motion_intensity] += 1

        dominant_layouts = [l for l, _ in layout_counter.most_common(3)]
        dominant_motion = motion_counter.most_common(1)[0][0]
        visual_confidence = (
            sum(1 for ext in extractions if ext.structured_metadata.visual.layout_pattern == dominant_layouts[0])
            / len(extractions)
        ) if dominant_layouts else 0.0
        confidence_scores["visual"] = visual_confidence

        visual_style = BrandVisualStyle(
            dominant_colors=dominant_colors,
            color_consistency=color_confidence,
            typography_styles=dominant_styles,
            layout_patterns=dominant_layouts,
            motion_style=dominant_motion,
            confidence=visual_confidence,
        )

        # Visual fingerprint — centroid of embeddings, L2-normalized
        embeddings = np.array([ext.embedding for ext in extractions])
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        fingerprint = centroid.tolist()

        # Outlier detection
        for ext in extractions:
            emb = np.array(ext.embedding)
            emb_norm = np.linalg.norm(emb)
            if emb_norm > 0:
                sim = float(np.dot(centroid, emb / emb_norm))
                if sim < 0.7:
                    logger.warning(
                        "brand_outlier_detected",
                        asset=ext.asset_path,
                        similarity=f"{sim:.3f}",
                    )

        return rules, voice, visual_style, fingerprint, confidence_scores
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_pattern_aggregator.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff format src/omni_proof/brand_extraction/pattern_aggregator.py`

- [ ] **Step 6: Commit**

```bash
git add src/omni_proof/brand_extraction/pattern_aggregator.py tests/unit/test_pattern_aggregator.py
git commit -m "feat: add PatternAggregator for brand pattern extraction"
```

---

## Task 3: ConflictDetector

**Files:**
- Create: `src/omni_proof/brand_extraction/conflict_detector.py`
- Test: `tests/unit/test_conflict_detector.py`

Pure logic. Compares new patterns against existing profile to find conflicts.

- [ ] **Step 1: Write tests**

Create `tests/unit/test_conflict_detector.py`:

```python
"""Tests for brand conflict detection."""

from datetime import datetime, timezone

import pytest

from omni_proof.brand_extraction.conflict_detector import ConflictDetector
from omni_proof.brand_extraction.models import (
    BrandProfile,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule


def _make_profile(**overrides) -> BrandProfile:
    now = datetime.now(timezone.utc)
    defaults = dict(
        profile_id="bp-1",
        brand_name="TestBrand",
        rules=[
            BrandRule(
                rule_id="r1",
                section_type="color_palette",
                description="Blue and white",
                hex_codes=["#004E89", "#FFFFFF"],
            ),
            BrandRule(
                rule_id="r2",
                section_type="typography",
                description="Sans-serif",
                approved_fonts=["Helvetica"],
            ),
        ],
        voice=BrandVoice(
            formality="formal",
            emotional_register="authoritative",
            vocabulary_themes=["reliability"],
            sentence_style="mixed",
            confidence=0.9,
        ),
        visual_style=BrandVisualStyle(
            dominant_colors=["#004E89", "#FFFFFF"],
            color_consistency=0.9,
            typography_styles=["sans-serif"],
            layout_patterns=["centered"],
            motion_style="static",
            confidence=0.8,
        ),
        visual_fingerprint=[0.1] * 10,
        source_assets=["/tmp/ad1.jpg"],
        extractions=[],
        confidence_scores={"color": 0.9},
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)
    return BrandProfile(**defaults)


class TestConflictDetector:
    def setup_method(self):
        self.detector = ConflictDetector()

    def test_no_conflicts_when_consistent(self):
        profile = _make_profile()
        new_rules = [
            BrandRule(
                rule_id="r3",
                section_type="color_palette",
                description="Blue and white",
                hex_codes=["#004E89", "#FFFFFF"],
            ),
        ]
        new_voice = BrandVoice(
            formality="formal",
            emotional_register="authoritative",
            vocabulary_themes=["reliability"],
            sentence_style="mixed",
            confidence=0.9,
        )
        new_visual = BrandVisualStyle(
            dominant_colors=["#004E89", "#FFFFFF"],
            color_consistency=0.9,
            typography_styles=["sans-serif"],
            layout_patterns=["centered"],
            motion_style="static",
            confidence=0.8,
        )
        conflicts = self.detector.detect(profile, new_rules, new_voice, new_visual)
        assert len(conflicts) == 0

    def test_major_color_conflict(self):
        profile = _make_profile()
        new_rules = [
            BrandRule(
                rule_id="r3",
                section_type="color_palette",
                description="Red palette",
                hex_codes=["#FF0000", "#CC0000"],
            ),
        ]
        new_voice = profile.voice
        new_visual = BrandVisualStyle(
            dominant_colors=["#FF0000", "#CC0000"],
            color_consistency=0.9,
            typography_styles=["sans-serif"],
            layout_patterns=["centered"],
            motion_style="static",
            confidence=0.8,
        )
        conflicts = self.detector.detect(profile, new_rules, new_voice, new_visual)
        color_conflicts = [c for c in conflicts if c.dimension == "color_palette"]
        assert len(color_conflicts) > 0
        assert color_conflicts[0].severity == "major"

    def test_major_voice_conflict(self):
        profile = _make_profile()
        new_rules = profile.rules
        new_voice = BrandVoice(
            formality="casual",
            emotional_register="playful",
            vocabulary_themes=["fun"],
            sentence_style="short_punchy",
            confidence=0.8,
        )
        new_visual = profile.visual_style
        conflicts = self.detector.detect(profile, new_rules, new_voice, new_visual)
        voice_conflicts = [c for c in conflicts if c.dimension == "tone"]
        assert len(voice_conflicts) > 0
        assert any(c.severity == "major" for c in voice_conflicts)

    def test_minor_typography_addition(self):
        profile = _make_profile()
        new_rules = [
            BrandRule(
                rule_id="r3",
                section_type="typography",
                description="Sans-serif and serif",
                approved_fonts=["Helvetica", "Georgia"],
            ),
        ]
        new_voice = profile.voice
        new_visual = BrandVisualStyle(
            dominant_colors=["#004E89", "#FFFFFF"],
            color_consistency=0.9,
            typography_styles=["sans-serif", "serif"],
            layout_patterns=["centered"],
            motion_style="static",
            confidence=0.8,
        )
        conflicts = self.detector.detect(profile, new_rules, new_voice, new_visual)
        typo_conflicts = [c for c in conflicts if c.dimension == "typography"]
        assert len(typo_conflicts) > 0
        assert typo_conflicts[0].severity == "major"
```

- [ ] **Step 2: Run tests to see them fail**

Run: `.venv/bin/pytest tests/unit/test_conflict_detector.py -v --tb=short`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement ConflictDetector**

Create `src/omni_proof/brand_extraction/conflict_detector.py`:

```python
"""Detects conflicts between new brand patterns and existing profile."""

import structlog

from omni_proof.brand_extraction.models import (
    BrandConflict,
    BrandProfile,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule

logger = structlog.get_logger()


class ConflictDetector:
    """Compares new aggregated patterns against an existing BrandProfile."""

    def detect(
        self,
        existing_profile: BrandProfile,
        new_rules: list[BrandRule],
        new_voice: BrandVoice,
        new_visual_style: BrandVisualStyle,
    ) -> list[BrandConflict]:
        conflicts: list[BrandConflict] = []

        conflicts.extend(self._check_colors(existing_profile, new_visual_style))
        conflicts.extend(self._check_typography(existing_profile, new_visual_style))
        conflicts.extend(self._check_voice(existing_profile, new_voice))
        conflicts.extend(self._check_visual(existing_profile, new_visual_style))

        logger.info("conflict_detection_complete", conflicts=len(conflicts))
        return conflicts

    def _check_colors(
        self, profile: BrandProfile, new_visual: BrandVisualStyle
    ) -> list[BrandConflict]:
        existing_colors = set(profile.visual_style.dominant_colors)
        new_colors = set(new_visual.dominant_colors)

        if not new_colors or new_colors <= existing_colors:
            return []

        novel = new_colors - existing_colors
        overlap = new_colors & existing_colors

        if not overlap:
            severity = "major"
        else:
            severity = "minor"

        return [
            BrandConflict(
                dimension="color_palette",
                existing_value=", ".join(sorted(existing_colors)),
                new_value=", ".join(sorted(novel)),
                source_assets=[],
                severity=severity,
            )
        ]

    def _check_typography(
        self, profile: BrandProfile, new_visual: BrandVisualStyle
    ) -> list[BrandConflict]:
        existing_styles = set(profile.visual_style.typography_styles)
        new_styles = set(new_visual.typography_styles)

        if not new_styles or new_styles <= existing_styles:
            return []

        novel = new_styles - existing_styles
        return [
            BrandConflict(
                dimension="typography",
                existing_value=", ".join(sorted(existing_styles)),
                new_value=", ".join(sorted(novel)),
                source_assets=[],
                severity="major",
            )
        ]

    def _check_voice(
        self, profile: BrandProfile, new_voice: BrandVoice
    ) -> list[BrandConflict]:
        conflicts = []

        if new_voice.formality != profile.voice.formality:
            conflicts.append(
                BrandConflict(
                    dimension="tone",
                    existing_value=f"formality={profile.voice.formality}",
                    new_value=f"formality={new_voice.formality}",
                    source_assets=[],
                    severity="major",
                )
            )

        if new_voice.emotional_register != profile.voice.emotional_register:
            conflicts.append(
                BrandConflict(
                    dimension="tone",
                    existing_value=f"register={profile.voice.emotional_register}",
                    new_value=f"register={new_voice.emotional_register}",
                    source_assets=[],
                    severity="major",
                )
            )

        return conflicts

    def _check_visual(
        self, profile: BrandProfile, new_visual: BrandVisualStyle
    ) -> list[BrandConflict]:
        conflicts = []

        if (
            new_visual.motion_style != profile.visual_style.motion_style
            and profile.visual_style.motion_style != "static"
        ):
            conflicts.append(
                BrandConflict(
                    dimension="visual_style",
                    existing_value=f"motion={profile.visual_style.motion_style}",
                    new_value=f"motion={new_visual.motion_style}",
                    source_assets=[],
                    severity="minor",
                )
            )

        return conflicts
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_conflict_detector.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/omni_proof/brand_extraction/conflict_detector.py tests/unit/test_conflict_detector.py
git commit -m "feat: add ConflictDetector for brand update conflict detection"
```

---

## Task 4: AssetProcessor

**Files:**
- Create: `src/omni_proof/brand_extraction/asset_processor.py`
- Test: `tests/unit/test_asset_processor.py`

Processes individual assets through Gemini Flash + Embedding 2. Requires mocking external calls.

- [ ] **Step 1: Write tests**

Create `tests/unit/test_asset_processor.py`:

```python
"""Tests for brand asset processing."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from omni_proof.brand_extraction.asset_processor import AssetProcessor
from omni_proof.brand_extraction.models import BrandAssetMetadata, BrandColorInfo, BrandToneInfo, BrandTypographyInfo, BrandVisualInfo
from omni_proof.core.exceptions import IngestionError


def _mock_metadata():
    return BrandAssetMetadata(
        asset_description="Test ad",
        colors=BrandColorInfo(hex_codes=["#004E89"], palette_mood="cool"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"], font_names=["Helvetica"], text_hierarchy="flat"
        ),
        tone=BrandToneInfo(
            formality="formal",
            emotional_register="authoritative",
            key_phrases=[],
            vocabulary_themes=["reliability"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered", motion_intensity="static", dominant_objects=["logo"]
        ),
        logo_detected=True,
        media_type_detected="image",
    )


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.extract_metadata = AsyncMock(return_value=_mock_metadata())
    return client


@pytest.fixture
def mock_embedding_provider():
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return provider


@pytest.fixture
def processor(mock_embedding_provider, mock_gemini):
    return AssetProcessor(
        embedding_provider=mock_embedding_provider,
        gemini_client=mock_gemini,
    )


class TestAssetProcessor:
    @pytest.mark.asyncio
    async def test_process_single_asset(self, processor, mock_gemini, mock_embedding_provider):
        result = await processor.process(Path("/tmp/ad.jpg"))

        assert result.asset_path == "/tmp/ad.jpg"
        assert result.media_type == "image"
        assert len(result.embedding) == 3072
        assert result.structured_metadata.colors.hex_codes == ["#004E89"]
        mock_gemini.extract_metadata.assert_awaited_once()
        mock_embedding_provider.generate_embedding.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_detects_media_type(self, processor):
        for ext, expected in [(".jpg", "image"), (".mp4", "video"), (".mp3", "audio"), (".pdf", "pdf")]:
            result = await processor.process(Path(f"/tmp/asset{ext}"))
            assert result.media_type == expected

    @pytest.mark.asyncio
    async def test_process_batch_skips_failures(self, processor, mock_gemini):
        call_count = 0

        async def failing_extract(path, schema):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Gemini error")
            return _mock_metadata()

        mock_gemini.extract_metadata = AsyncMock(side_effect=failing_extract)

        results = await processor.process_batch([
            Path("/tmp/ad1.jpg"),
            Path("/tmp/ad2.jpg"),
            Path("/tmp/ad3.jpg"),
        ])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_process_batch_all_fail_raises(self, processor, mock_gemini):
        mock_gemini.extract_metadata = AsyncMock(side_effect=RuntimeError("fail"))

        with pytest.raises(IngestionError, match="All .* assets failed"):
            await processor.process_batch([Path("/tmp/ad1.jpg"), Path("/tmp/ad2.jpg")])

    @pytest.mark.asyncio
    async def test_embedding_uses_task_type(self, processor, mock_embedding_provider):
        await processor.process(Path("/tmp/ad.jpg"))
        call_kwargs = mock_embedding_provider.generate_embedding.call_args.kwargs
        assert call_kwargs.get("task_type") == "SEMANTIC_SIMILARITY"
```

- [ ] **Step 2: Run tests to see them fail**

Run: `.venv/bin/pytest tests/unit/test_asset_processor.py -v --tb=short`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement AssetProcessor**

Create `src/omni_proof/brand_extraction/asset_processor.py`:

```python
"""Processes individual assets through Gemini Flash and Embedding 2."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import structlog

from omni_proof.brand_extraction.models import AssetExtraction, BrandAssetMetadata
from omni_proof.core.exceptions import IngestionError
from omni_proof.core.interfaces import EmbeddingProvider
from omni_proof.ingestion.gemini_client import GeminiClient

logger = structlog.get_logger()

MEDIA_TYPE_MAP = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image",
    ".mp4": "video", ".avi": "video", ".mov": "video", ".webm": "video",
    ".mp3": "audio", ".wav": "audio", ".ogg": "audio", ".flac": "audio",
    ".pdf": "pdf",
}


class AssetProcessor:
    """Processes brand assets through Gemini for structured extraction and embedding."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        gemini_client: GeminiClient,
    ):
        self._embedding = embedding_provider
        self._gemini = gemini_client

    def _detect_media_type(self, path: Path) -> str:
        return MEDIA_TYPE_MAP.get(path.suffix.lower(), "image")

    async def process(self, asset_path: Path) -> AssetExtraction:
        media_type = self._detect_media_type(asset_path)

        metadata, embedding = await asyncio.gather(
            self._gemini.extract_metadata(asset_path, BrandAssetMetadata),
            self._embedding.generate_embedding(
                asset_path, task_type="SEMANTIC_SIMILARITY"
            ),
        )

        return AssetExtraction(
            asset_path=str(asset_path),
            media_type=media_type,
            embedding=embedding,
            structured_metadata=metadata,
            extracted_at=datetime.now(timezone.utc),
        )

    async def process_batch(self, assets: list[Path]) -> list[AssetExtraction]:
        results: list[AssetExtraction] = []
        for path in assets:
            try:
                result = await self.process(path)
                results.append(result)
            except Exception:
                logger.exception("brand_asset_processing_failed", path=str(path))

        if not results:
            raise IngestionError(f"All {len(assets)} assets failed during brand extraction")

        return results
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_asset_processor.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/omni_proof/brand_extraction/asset_processor.py tests/unit/test_asset_processor.py
git commit -m "feat: add AssetProcessor for brand asset ingestion"
```

---

## Task 5: BrandExtractor Orchestrator

**Files:**
- Create: `src/omni_proof/brand_extraction/extractor.py`
- Test: `tests/unit/test_brand_extractor.py`

The orchestrator that ties all components together.

- [ ] **Step 1: Write tests**

Create `tests/unit/test_brand_extractor.py`:

```python
"""Tests for BrandExtractor orchestrator."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandConflict,
    BrandProfile,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
    BrandVisualStyle,
    BrandVoice,
)
from omni_proof.rag.models import BrandRule


def _make_extraction(path: str = "/tmp/ad.jpg") -> AssetExtraction:
    return AssetExtraction(
        asset_path=path,
        media_type="image",
        embedding=[0.1] * 10,
        structured_metadata=BrandAssetMetadata(
            asset_description="Test",
            colors=BrandColorInfo(hex_codes=["#004E89"], palette_mood="cool"),
            typography=BrandTypographyInfo(
                font_styles=["sans-serif"], font_names=["Helvetica"], text_hierarchy="flat"
            ),
            tone=BrandToneInfo(
                formality="formal", emotional_register="authoritative",
                key_phrases=[], vocabulary_themes=["reliability"],
            ),
            visual=BrandVisualInfo(
                layout_pattern="centered", motion_intensity="static", dominant_objects=["logo"]
            ),
            logo_detected=True,
            media_type_detected="image",
        ),
        extracted_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def mock_deps():
    embedding = AsyncMock()
    embedding.generate_embedding = AsyncMock(return_value=[0.1] * 10)
    gemini = AsyncMock()
    vector_store = AsyncMock()
    vector_store.upsert = AsyncMock()
    return embedding, gemini, vector_store


class TestBrandExtractorExtract:
    @pytest.mark.asyncio
    async def test_extract_returns_brand_profile(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)

        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(return_value=[
                _make_extraction("/tmp/a.jpg"),
                _make_extraction("/tmp/b.jpg"),
            ])
            profile = await extractor.extract("TestBrand", [Path("/tmp/a.jpg"), Path("/tmp/b.jpg")])

        assert profile.brand_name == "TestBrand"
        assert len(profile.source_assets) == 2
        assert len(profile.rules) > 0
        assert profile.voice.formality == "formal"
        assert len(profile.visual_fingerprint) == 10
        assert len(profile.extractions) == 2

    @pytest.mark.asyncio
    async def test_extract_indexes_into_vector_store(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)

        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(return_value=[_make_extraction()])
            await extractor.extract("TestBrand", [Path("/tmp/a.jpg")])

        # Should index rules + fingerprint into vector store
        assert store.upsert.await_count > 0


class TestBrandExtractorUpdate:
    @pytest.mark.asyncio
    async def test_update_detects_conflicts(self, mock_deps):
        embedding, gemini, store = mock_deps
        extractor = BrandExtractor(embedding, gemini, store)

        now = datetime.now(timezone.utc)
        existing = BrandProfile(
            profile_id="bp-1",
            brand_name="TestBrand",
            rules=[BrandRule(rule_id="r1", section_type="color_palette", description="Blue", hex_codes=["#004E89"])],
            voice=BrandVoice(formality="formal", emotional_register="authoritative", vocabulary_themes=["reliability"], sentence_style="mixed", confidence=0.9),
            visual_style=BrandVisualStyle(dominant_colors=["#004E89"], color_consistency=0.9, typography_styles=["sans-serif"], layout_patterns=["centered"], motion_style="static", confidence=0.8),
            visual_fingerprint=[0.1] * 10,
            source_assets=["/tmp/old.jpg"],
            extractions=[_make_extraction("/tmp/old.jpg")],
            confidence_scores={"color": 0.9},
            created_at=now,
            updated_at=now,
        )

        # New asset has different voice
        new_ext = _make_extraction("/tmp/new.jpg")
        new_ext.structured_metadata.tone.formality = "casual"
        new_ext.structured_metadata.tone.emotional_register = "playful"

        with patch.object(extractor, "_processor") as mock_proc:
            mock_proc.process_batch = AsyncMock(return_value=[new_ext])
            updated, conflicts = await extractor.update(existing, [Path("/tmp/new.jpg")])

        assert updated.brand_name == "TestBrand"
        assert len(updated.source_assets) == 2
        assert len(updated.extractions) == 2
        assert len(conflicts) > 0
```

- [ ] **Step 2: Run tests to see them fail**

Run: `.venv/bin/pytest tests/unit/test_brand_extractor.py -v --tb=short`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement BrandExtractor**

Create `src/omni_proof/brand_extraction/extractor.py`:

```python
"""BrandExtractor orchestrator — ties asset processing, aggregation, and conflict detection."""

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
    """Orchestrates brand identity extraction from multimodal assets."""

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

    async def extract(
        self,
        brand_name: str,
        assets: list[Path],
    ) -> BrandProfile:
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

        logger.info(
            "brand_extracted",
            brand=brand_name,
            rules=len(rules),
            assets=len(assets),
        )
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

        conflicts = self._conflict_detector.detect(
            profile, rules, voice, visual_style
        )

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
        # Index visual fingerprint as a special vector
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

        # Index each rule description as an embedding for RAG retrieval
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
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_brand_extractor.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/omni_proof/brand_extraction/extractor.py tests/unit/test_brand_extractor.py
git commit -m "feat: add BrandExtractor orchestrator"
```

---

## Task 6: API Routes

**Files:**
- Create: `src/omni_proof/api/routes/brand.py`
- Modify: `src/omni_proof/api/app.py`
- Test: `tests/unit/test_api_routes.py` (add brand route tests)

- [ ] **Step 1: Create brand API routes**

Create `src/omni_proof/api/routes/brand.py`:

```python
"""Brand extraction API routes."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from omni_proof.api.deps import get_settings
from omni_proof.brand_extraction.models import BrandConflict, BrandProfile
from omni_proof.config.settings import Settings

router = APIRouter()


class ExtractRequest(BaseModel):
    brand_name: str
    asset_paths: list[str]


class BrandProfileResponse(BaseModel):
    profile: BrandProfile
    outlier_assets: list[str] = []


class BrandUpdateResponse(BaseModel):
    updated_profile: BrandProfile
    conflicts: list[BrandConflict]
    outlier_assets: list[str] = []


@router.post("/extract")
async def extract_brand(
    request: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    return {
        "status": "not_configured",
        "message": "Configure embedding provider and Gemini client to run brand extraction",
        "brand_name": request.brand_name,
        "asset_count": len(request.asset_paths),
    }


@router.get("/profile/{profile_id}")
async def get_profile(profile_id: str) -> dict:
    return {
        "status": "not_configured",
        "message": "Configure storage to retrieve brand profiles",
        "profile_id": profile_id,
    }


@router.post("/update/{profile_id}")
async def update_brand(
    profile_id: str,
    request: ExtractRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    return {
        "status": "not_configured",
        "message": "Configure embedding provider and Gemini client to update brand profiles",
        "profile_id": profile_id,
        "brand_name": request.brand_name,
        "new_asset_count": len(request.asset_paths),
    }
```

- [ ] **Step 2: Wire routes into app**

In `src/omni_proof/api/app.py`, add import and router:

```python
from omni_proof.api.routes import causal, compliance, insights, generative, brand
```

And in `create_app()`:
```python
app.include_router(brand.router, prefix="/api/v1/brand", tags=["brand"])
```

- [ ] **Step 3: Add tests**

Add to `tests/unit/test_api_routes.py`:

```python
class TestBrandRoutes:
    def test_extract_brand(self, client):
        resp = client.post(
            "/api/v1/brand/extract",
            json={"brand_name": "TestBrand", "asset_paths": ["/tmp/a.jpg"]},
        )
        assert resp.status_code == 200
        assert resp.json()["brand_name"] == "TestBrand"
        assert resp.json()["asset_count"] == 1

    def test_get_profile(self, client):
        resp = client.get("/api/v1/brand/profile/bp-1")
        assert resp.status_code == 200
        assert resp.json()["profile_id"] == "bp-1"

    def test_update_brand(self, client):
        resp = client.post(
            "/api/v1/brand/update/bp-1",
            json={"brand_name": "TestBrand", "asset_paths": ["/tmp/new.jpg"]},
        )
        assert resp.status_code == 200
        assert resp.json()["new_asset_count"] == 1
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/unit/test_api_routes.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/omni_proof/api/routes/brand.py src/omni_proof/api/app.py tests/unit/test_api_routes.py
git commit -m "feat: add brand extraction API routes"
```

---

## Task 7: Public Exports + Integration Test

**Files:**
- Modify: `src/omni_proof/brand_extraction/__init__.py`
- Modify: `src/omni_proof/__init__.py`
- Create: `tests/integration/test_brand_extraction_e2e.py`

- [ ] **Step 1: Add exports**

Update `src/omni_proof/brand_extraction/__init__.py`:

```python
from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import BrandProfile

__all__ = ["BrandExtractor", "BrandProfile"]
```

Add to `src/omni_proof/__init__.py` imports and `__all__`:

```python
from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import BrandProfile
```

Add `"BrandExtractor"` and `"BrandProfile"` to the `__all__` list.

- [ ] **Step 2: Write integration test**

Create `tests/integration/test_brand_extraction_e2e.py`:

```python
"""End-to-end test for brand extraction pipeline."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.brand_extraction.extractor import BrandExtractor
from omni_proof.brand_extraction.models import (
    BrandAssetMetadata,
    BrandColorInfo,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
)
from omni_proof.storage.memory_store import InMemoryVectorStore


def _make_mock_metadata(**overrides):
    defaults = dict(
        asset_description="Brand ad",
        colors=BrandColorInfo(hex_codes=["#004E89", "#FFFFFF"], palette_mood="cool"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"], font_names=["Helvetica"], text_hierarchy="strong_hierarchy"
        ),
        tone=BrandToneInfo(
            formality="formal", emotional_register="authoritative",
            key_phrases=["Trust the process"], vocabulary_themes=["reliability", "innovation"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered", motion_intensity="static", dominant_objects=["logo", "product"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )
    defaults.update(overrides)
    return BrandAssetMetadata(**defaults)


@pytest.fixture
def mock_gemini():
    client = AsyncMock()
    client.extract_metadata = AsyncMock(return_value=_make_mock_metadata())
    return client


@pytest.fixture
def mock_embedding():
    provider = AsyncMock()
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 3072)
    return provider


@pytest.fixture
def memory_store():
    return InMemoryVectorStore()


class TestBrandExtractionE2E:
    @pytest.mark.asyncio
    async def test_full_extract_and_update_cycle(self, mock_gemini, mock_embedding, memory_store):
        extractor = BrandExtractor(mock_embedding, mock_gemini, memory_store)

        # Extract from initial assets
        profile = await extractor.extract(
            "AcmeCorp",
            [Path("/tmp/ad1.jpg"), Path("/tmp/ad2.jpg"), Path("/tmp/guide.pdf")],
        )

        assert profile.brand_name == "AcmeCorp"
        assert len(profile.rules) > 0
        assert "#004E89" in profile.visual_style.dominant_colors
        assert profile.voice.formality == "formal"
        assert len(profile.extractions) == 3

        # Verify indexed into vector store
        results = await memory_store.search(
            [0.1] * 3072, top_k=10, namespace="brand_assets"
        )
        assert len(results) > 0

        # Update with new assets that have same brand patterns — no conflicts
        updated, conflicts = await extractor.update(
            profile, [Path("/tmp/new_ad.jpg")]
        )
        assert len(updated.extractions) == 4
        assert len(updated.source_assets) == 4
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_update_detects_conflicts(self, mock_gemini, mock_embedding, memory_store):
        extractor = BrandExtractor(mock_embedding, mock_gemini, memory_store)

        profile = await extractor.extract("AcmeCorp", [Path("/tmp/ad1.jpg")])

        # New asset with conflicting voice
        mock_gemini.extract_metadata = AsyncMock(
            return_value=_make_mock_metadata(
                tone=BrandToneInfo(
                    formality="casual",
                    emotional_register="playful",
                    key_phrases=["Let's go!"],
                    vocabulary_themes=["fun", "adventure"],
                ),
            )
        )

        updated, conflicts = await extractor.update(profile, [Path("/tmp/rebrand.jpg")])
        assert len(conflicts) > 0
        tone_conflicts = [c for c in conflicts if c.dimension == "tone"]
        assert len(tone_conflicts) > 0
```

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/pytest tests/ -v --tb=short`
Expected: All pass (103 existing + new brand extraction tests).

- [ ] **Step 4: Verify imports**

Run: `.venv/bin/python -c "from omni_proof import BrandExtractor, BrandProfile; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Format and lint**

Run: `.venv/bin/ruff format src/omni_proof/brand_extraction/ tests/unit/test_brand_extraction_models.py tests/unit/test_pattern_aggregator.py tests/unit/test_conflict_detector.py tests/unit/test_asset_processor.py tests/unit/test_brand_extractor.py tests/integration/test_brand_extraction_e2e.py`
Run: `.venv/bin/ruff check src/omni_proof/brand_extraction/ --select E,F,I,W`

- [ ] **Step 6: Commit**

```bash
git add src/omni_proof/brand_extraction/__init__.py src/omni_proof/__init__.py tests/integration/test_brand_extraction_e2e.py
git commit -m "feat: add brand extraction public exports and integration test"
```

---

## Verification

After all tasks complete:

```bash
# All tests pass
.venv/bin/pytest tests/ -v --tb=short

# Imports work
.venv/bin/python -c "from omni_proof import BrandExtractor, BrandProfile, DMLEstimator; print('OK')"

# Lint passes
.venv/bin/ruff check src/ tests/ --select E,F,I,W

# API starts
uvicorn omni_proof.api.app:create_app --factory &
curl localhost:8000/health
curl -X POST localhost:8000/api/v1/brand/extract -H "Content-Type: application/json" -d '{"brand_name":"Test","asset_paths":["/tmp/a.jpg"]}'
kill %1
```

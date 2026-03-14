"""Unit tests for PatternAggregator."""

from datetime import datetime

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
    asset_path: str = "test_asset.jpg",
    media_type: str = "image",
    embedding: list[float] | None = None,
    hex_codes: list[str] | None = None,
    font_styles: list[str] | None = None,
    font_names: list[str] | None = None,
    formality: str = "formal",
    emotional_register: str = "professional",
    vocabulary_themes: list[str] | None = None,
    key_phrases: list[str] | None = None,
    layout_pattern: str = "grid",
    motion_intensity: str = "low",
) -> AssetExtraction:
    """Helper to create AssetExtraction with customizable fields and sensible defaults."""
    if embedding is None:
        # Create a random normalized embedding
        emb = np.random.randn(128)
        embedding = (emb / np.linalg.norm(emb)).tolist()

    if hex_codes is None:
        hex_codes = ["#000000", "#FFFFFF"]

    if font_styles is None:
        font_styles = ["sans-serif"]

    if font_names is None:
        font_names = ["Arial"]

    if vocabulary_themes is None:
        vocabulary_themes = ["business", "professional"]

    if key_phrases is None:
        key_phrases = ["conversational"]

    metadata = BrandAssetMetadata(
        asset_description="Test asset",
        colors=BrandColorInfo(hex_codes=hex_codes, palette_mood="neutral"),
        typography=BrandTypographyInfo(
            font_styles=font_styles, font_names=font_names, text_hierarchy="standard"
        ),
        tone=BrandToneInfo(
            formality=formality,
            emotional_register=emotional_register,
            key_phrases=key_phrases,
            vocabulary_themes=vocabulary_themes,
        ),
        visual=BrandVisualInfo(
            layout_pattern=layout_pattern,
            motion_intensity=motion_intensity,
            dominant_objects=["logo"],
        ),
        logo_detected=True,
        media_type_detected=media_type,
    )

    return AssetExtraction(
        asset_path=asset_path,
        media_type=media_type,
        embedding=embedding,
        structured_metadata=metadata,
        extracted_at=datetime.now(),
    )


def test_empty_extractions_raises():
    """Test that empty extractions list raises ValueError."""
    aggregator = PatternAggregator()
    with pytest.raises(ValueError, match="Cannot aggregate empty extractions list"):
        aggregator.aggregate([])


def test_single_asset_extraction():
    """Test aggregation works with a single asset."""
    aggregator = PatternAggregator()
    extraction = _make_extraction(
        asset_path="asset1.jpg",
        hex_codes=["#FF0000", "#00FF00"],
        font_styles=["serif"],
        font_names=["Georgia"],
    )

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate([extraction])

    # Should return rules
    assert len(rules) == 2
    assert any(rule.section_type == "typography" for rule in rules)
    assert any(rule.section_type == "color_palette" for rule in rules)

    # Typography rule should have fonts
    typo_rule = next(rule for rule in rules if rule.section_type == "typography")
    assert "Georgia" in typo_rule.approved_fonts

    # Color rule should have hex codes
    color_rule = next(rule for rule in rules if rule.section_type == "color_palette")
    assert "#FF0000" in color_rule.hex_codes

    # Voice should be extracted
    assert voice.formality == "formal"
    assert voice.emotional_register == "professional"
    assert voice.confidence == 1.0  # Single asset = 100% match

    # Visual style should be extracted
    assert "#FF0000" in visual.dominant_colors
    assert "serif" in visual.typography_styles

    # Fingerprint should be normalized
    fp_norm = np.linalg.norm(fingerprint)
    assert abs(fp_norm - 1.0) < 1e-6

    # Confidence scores
    assert "color" in confidence
    assert "typography" in confidence
    assert "voice" in confidence
    assert "visual" in confidence


def test_color_frequency_aggregation():
    """Test that most frequent color appears first in dominant_colors."""
    aggregator = PatternAggregator()

    # Asset 1: red, blue
    # Asset 2: red, green
    # Asset 3: red, yellow
    # Red should be most frequent
    extractions = [
        _make_extraction(asset_path="asset1.jpg", hex_codes=["#FF0000", "#0000FF"]),
        _make_extraction(asset_path="asset2.jpg", hex_codes=["#FF0000", "#00FF00"]),
        _make_extraction(asset_path="asset3.jpg", hex_codes=["#FF0000", "#FFFF00"]),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Red should be first in dominant colors
    assert visual.dominant_colors[0] == "#FF0000"
    assert len(visual.dominant_colors) <= 5

    # Color confidence: all 3 assets contain red (top color)
    assert confidence["color"] == 1.0


def test_voice_consistency():
    """Test voice aggregation with majority voting."""
    aggregator = PatternAggregator()

    # 2 formal, 1 casual
    extractions = [
        _make_extraction(
            asset_path="asset1.jpg",
            formality="formal",
            emotional_register="professional",
        ),
        _make_extraction(
            asset_path="asset2.jpg",
            formality="formal",
            emotional_register="professional",
        ),
        _make_extraction(
            asset_path="asset3.jpg",
            formality="casual",
            emotional_register="friendly",
        ),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Majority wins
    assert voice.formality == "formal"
    assert voice.emotional_register == "professional"

    # Confidence should be ~0.67 (2 out of 3)
    assert abs(voice.confidence - 2 / 3) < 0.01


def test_visual_fingerprint_is_normalized():
    """Test that visual fingerprint is L2-normalized."""
    aggregator = PatternAggregator()

    # Create 3 assets with different embeddings
    extractions = [
        _make_extraction(asset_path=f"asset{i}.jpg", embedding=(np.random.randn(128) * 10).tolist())
        for i in range(3)
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Fingerprint should be normalized
    fp_norm = np.linalg.norm(fingerprint)
    assert abs(fp_norm - 1.0) < 1e-6


def test_typography_aggregation():
    """Test that most common font style appears in typography_styles."""
    aggregator = PatternAggregator()

    # 3 assets: 2 with serif, 1 with sans-serif
    extractions = [
        _make_extraction(
            asset_path="asset1.jpg",
            font_styles=["serif", "bold"],
            font_names=["Georgia"],
        ),
        _make_extraction(
            asset_path="asset2.jpg",
            font_styles=["serif", "italic"],
            font_names=["Times New Roman"],
        ),
        _make_extraction(
            asset_path="asset3.jpg",
            font_styles=["sans-serif"],
            font_names=["Arial"],
        ),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Serif should be most common
    assert "serif" in visual.typography_styles
    assert visual.typography_styles[0] == "serif"

    # Typography rule should have the fonts
    typo_rule = next(rule for rule in rules if rule.section_type == "typography")
    assert len(typo_rule.approved_fonts) > 0
    assert "Georgia" in typo_rule.approved_fonts or "Times New Roman" in typo_rule.approved_fonts


def test_vocabulary_themes_deduplication():
    """Test that vocabulary themes are deduplicated and limited to 20."""
    aggregator = PatternAggregator()

    # Create extractions with overlapping themes
    extractions = [
        _make_extraction(
            asset_path="asset1.jpg",
            vocabulary_themes=["business", "professional", "corporate"],
        ),
        _make_extraction(
            asset_path="asset2.jpg",
            vocabulary_themes=["professional", "innovation", "technology"],
        ),
        _make_extraction(
            asset_path="asset3.jpg",
            vocabulary_themes=["business", "technology", "future"],
        ),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Should be deduplicated
    assert len(voice.vocabulary_themes) == len(set(voice.vocabulary_themes))
    # Should contain all unique themes
    assert "business" in voice.vocabulary_themes
    assert "professional" in voice.vocabulary_themes
    assert "innovation" in voice.vocabulary_themes
    # Max 20
    assert len(voice.vocabulary_themes) <= 20


def test_outlier_detection_logs_warning(caplog):
    """Test that outlier detection logs warnings for dissimilar assets."""
    import logging

    # Ensure the logger is capturing
    caplog.set_level(logging.WARNING, logger="omni_proof.brand_extraction.pattern_aggregator")

    aggregator = PatternAggregator()

    # Create 2 similar embeddings pointing in positive direction
    base_emb = np.ones(128)
    base_emb = base_emb / np.linalg.norm(base_emb)

    similar1 = base_emb.copy()
    similar2 = base_emb.copy()

    # Outlier: pointing in completely opposite direction (negative)
    outlier = -1.0 * base_emb  # This will have cosine similarity of -1 with base_emb

    extractions = [
        _make_extraction(asset_path="asset1.jpg", embedding=similar1.tolist()),
        _make_extraction(asset_path="asset2.jpg", embedding=similar2.tolist()),
        _make_extraction(asset_path="outlier.jpg", embedding=outlier.tolist()),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Should log warning for outlier
    assert any("outlier.jpg" in record.message for record in caplog.records)
    assert any("outlier" in record.message.lower() for record in caplog.records)


def test_confidence_scores_structure():
    """Test that confidence scores have all required keys."""
    aggregator = PatternAggregator()

    extractions = [_make_extraction(asset_path=f"asset{i}.jpg") for i in range(3)]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # All keys should be present
    assert "color" in confidence
    assert "typography" in confidence
    assert "voice" in confidence
    assert "visual" in confidence

    # All values should be between 0 and 1
    for key, value in confidence.items():
        assert 0.0 <= value <= 1.0


def test_layout_patterns_aggregation():
    """Test that layout patterns are aggregated correctly."""
    aggregator = PatternAggregator()

    # 2 grid, 1 centered
    extractions = [
        _make_extraction(asset_path="asset1.jpg", layout_pattern="grid"),
        _make_extraction(asset_path="asset2.jpg", layout_pattern="grid"),
        _make_extraction(asset_path="asset3.jpg", layout_pattern="centered"),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Grid should be most common
    assert "grid" in visual.layout_patterns
    assert visual.layout_patterns[0] == "grid"


def test_motion_style_aggregation():
    """Test that motion style uses most common value."""
    aggregator = PatternAggregator()

    # 2 low, 1 high
    extractions = [
        _make_extraction(asset_path="asset1.jpg", motion_intensity="low"),
        _make_extraction(asset_path="asset2.jpg", motion_intensity="low"),
        _make_extraction(asset_path="asset3.jpg", motion_intensity="high"),
    ]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # Low should win
    assert visual.motion_style == "low"


def test_rules_have_unique_ids():
    """Test that generated rules have unique UUIDs."""
    aggregator = PatternAggregator()

    extractions = [_make_extraction(asset_path="asset1.jpg")]

    rules, voice, visual, fingerprint, confidence = aggregator.aggregate(extractions)

    # All rule IDs should be unique
    rule_ids = [rule.rule_id for rule in rules]
    assert len(rule_ids) == len(set(rule_ids))

    # Should be valid UUIDs (basic check)
    for rule_id in rule_ids:
        assert len(rule_id) == 36  # UUID string length
        assert rule_id.count("-") == 4  # UUID format

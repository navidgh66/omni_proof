"""Unit tests for brand asset processor."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from omni_proof.brand_extraction.asset_processor import AssetProcessor
from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandAssetMetadata,
    BrandColorInfo,
    BrandToneInfo,
    BrandTypographyInfo,
    BrandVisualInfo,
)
from omni_proof.core.exceptions import IngestionError


@pytest.fixture
def mock_brand_metadata():
    """Create valid BrandAssetMetadata."""
    return BrandAssetMetadata(
        asset_description="Test asset",
        colors=BrandColorInfo(hex_codes=["#FFFFFF"], palette_mood="bright"),
        typography=BrandTypographyInfo(
            font_styles=["sans-serif"],
            font_names=["Arial"],
            text_hierarchy="title",
        ),
        tone=BrandToneInfo(
            formality="casual",
            emotional_register="upbeat",
            key_phrases=["hello"],
            vocabulary_themes=["greeting"],
        ),
        visual=BrandVisualInfo(
            layout_pattern="centered",
            motion_intensity="low",
            dominant_objects=["logo"],
        ),
        logo_detected=True,
        media_type_detected="image",
    )


@pytest.fixture
def mock_gemini(mock_brand_metadata):
    """Create mock GeminiClient."""
    client = AsyncMock()
    client.extract_metadata.return_value = mock_brand_metadata
    return client


@pytest.fixture
def mock_embedding_provider():
    """Create mock EmbeddingProvider."""
    provider = AsyncMock()
    provider.generate_embedding.return_value = [0.1] * 3072
    return provider


@pytest.fixture
def asset_processor(mock_embedding_provider, mock_gemini):
    """Create AssetProcessor with mocked dependencies."""
    return AssetProcessor(mock_embedding_provider, mock_gemini)


@pytest.mark.asyncio
async def test_process_single_asset(asset_processor, mock_brand_metadata):
    """Test processing a single asset returns AssetExtraction with correct fields."""
    asset_path = Path("/test/asset.jpg")

    result = await asset_processor.process(asset_path)

    assert isinstance(result, AssetExtraction)
    assert result.asset_path == str(asset_path)
    assert result.media_type == "image"
    assert len(result.embedding) == 3072
    assert result.embedding == [0.1] * 3072
    assert result.structured_metadata == mock_brand_metadata
    assert isinstance(result.extracted_at, datetime)
    assert result.extracted_at.tzinfo == UTC


@pytest.mark.asyncio
async def test_process_detects_media_type(asset_processor):
    """Test media type detection for different file extensions."""
    test_cases = [
        (Path("/test/image.jpg"), "image"),
        (Path("/test/image.jpeg"), "image"),
        (Path("/test/image.png"), "image"),
        (Path("/test/image.gif"), "image"),
        (Path("/test/image.webp"), "image"),
        (Path("/test/image.bmp"), "image"),
        (Path("/test/video.mp4"), "video"),
        (Path("/test/video.avi"), "video"),
        (Path("/test/video.mov"), "video"),
        (Path("/test/video.webm"), "video"),
        (Path("/test/audio.mp3"), "audio"),
        (Path("/test/audio.wav"), "audio"),
        (Path("/test/audio.ogg"), "audio"),
        (Path("/test/audio.flac"), "audio"),
        (Path("/test/document.pdf"), "pdf"),
        (Path("/test/unknown.xyz"), "image"),  # defaults to image
    ]

    for asset_path, expected_type in test_cases:
        result = await asset_processor.process(asset_path)
        assert result.media_type == expected_type, f"Failed for {asset_path.suffix}"


@pytest.mark.asyncio
async def test_process_batch_skips_failures(
    mock_embedding_provider, mock_gemini, mock_brand_metadata
):
    """Test batch processing continues when middle asset fails."""
    processor = AssetProcessor(mock_embedding_provider, mock_gemini)

    # Second call to extract_metadata will fail
    mock_gemini.extract_metadata.side_effect = [
        mock_brand_metadata,  # First asset succeeds
        Exception("Extraction failed"),  # Second asset fails
        mock_brand_metadata,  # Third asset succeeds
    ]

    assets = [Path("/test/asset1.jpg"), Path("/test/asset2.jpg"), Path("/test/asset3.jpg")]

    results = await processor.process_batch(assets)

    assert len(results) == 2
    assert results[0].asset_path == str(assets[0])
    assert results[1].asset_path == str(assets[2])


@pytest.mark.asyncio
async def test_process_batch_all_fail_raises(mock_embedding_provider, mock_gemini):
    """Test batch processing raises IngestionError when all assets fail."""
    processor = AssetProcessor(mock_embedding_provider, mock_gemini)

    # All calls fail
    mock_gemini.extract_metadata.side_effect = Exception("Extraction failed")

    assets = [Path("/test/asset1.jpg"), Path("/test/asset2.jpg")]

    with pytest.raises(IngestionError, match="All 2 assets failed during brand extraction"):
        await processor.process_batch(assets)


@pytest.mark.asyncio
async def test_embedding_uses_task_type(asset_processor, mock_embedding_provider):
    """Test that generate_embedding is called with task_type='SEMANTIC_SIMILARITY'."""
    asset_path = Path("/test/asset.jpg")

    await asset_processor.process(asset_path)

    mock_embedding_provider.generate_embedding.assert_called_once()
    call_args = mock_embedding_provider.generate_embedding.call_args
    assert call_args[0][0] == asset_path
    assert call_args[1]["task_type"] == "SEMANTIC_SIMILARITY"

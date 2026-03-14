"""Tests for asset preprocessor."""

from pathlib import Path

from omni_proof.ingestion.preprocessor import AssetPreprocessor


class TestImageBatching:
    def test_under_limit(self):
        paths = [Path(f"img_{i}.jpg") for i in range(4)]
        assert len(AssetPreprocessor().batch_images(paths)) == 1

    def test_over_limit(self):
        paths = [Path(f"img_{i}.jpg") for i in range(14)]
        batches = AssetPreprocessor().batch_images(paths)
        assert len(batches) == 3
        assert len(batches[0]) == 6
        assert len(batches[2]) == 2

    def test_empty(self):
        assert AssetPreprocessor().batch_images([]) == []

    def test_exact_multiple(self):
        paths = [Path(f"img_{i}.jpg") for i in range(12)]
        batches = AssetPreprocessor().batch_images(paths)
        assert len(batches) == 2
        assert all(len(b) == 6 for b in batches)


class TestPDFSegmentation:
    def test_under_limit(self):
        assert AssetPreprocessor().segment_pdf_by_count(4) == [(0, 4)]

    def test_over_limit(self):
        assert AssetPreprocessor().segment_pdf_by_count(15) == [(0, 6), (6, 12), (12, 15)]

    def test_exact_multiple(self):
        assert AssetPreprocessor().segment_pdf_by_count(12) == [(0, 6), (6, 12)]

    def test_zero_pages(self):
        assert AssetPreprocessor().segment_pdf_by_count(0) == []


class TestVideoChunking:
    def test_under_limit_with_audio(self):
        chunks = AssetPreprocessor().compute_video_chunks(60.0, has_audio=True)
        assert chunks == [(0.0, 60.0)]

    def test_over_limit_with_audio(self):
        chunks = AssetPreprocessor().compute_video_chunks(200.0, has_audio=True)
        assert chunks == [(0.0, 80.0), (80.0, 160.0), (160.0, 200.0)]

    def test_under_limit_no_audio(self):
        chunks = AssetPreprocessor().compute_video_chunks(100.0, has_audio=False)
        assert chunks == [(0.0, 100.0)]

    def test_over_limit_no_audio(self):
        chunks = AssetPreprocessor().compute_video_chunks(250.0, has_audio=False)
        assert chunks == [(0.0, 120.0), (120.0, 240.0), (240.0, 250.0)]

    def test_zero_duration(self):
        assert AssetPreprocessor().compute_video_chunks(0.0) == []


class TestAudioChunking:
    def test_under_limit(self):
        assert AssetPreprocessor().compute_audio_chunks(60.0) == [(0.0, 60.0)]

    def test_over_limit(self):
        chunks = AssetPreprocessor().compute_audio_chunks(200.0)
        assert chunks == [(0.0, 80.0), (80.0, 160.0), (160.0, 200.0)]

    def test_zero(self):
        assert AssetPreprocessor().compute_audio_chunks(0.0) == []

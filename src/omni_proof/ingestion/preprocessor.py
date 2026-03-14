"""Asset preprocessor for chunking/segmenting/batching media to fit Gemini API limits."""

from pathlib import Path

from omni_proof.config.constants import (
    MAX_AUDIO_SECONDS,
    MAX_IMAGES_PER_REQUEST,
    MAX_PDF_PAGES,
    MAX_VIDEO_SECONDS_NO_AUDIO,
    MAX_VIDEO_SECONDS_WITH_AUDIO,
)


class AssetPreprocessor:
    """Prepares raw assets for Gemini API ingestion."""

    def batch_images(
        self, image_paths: list[Path], batch_size: int = MAX_IMAGES_PER_REQUEST
    ) -> list[list[Path]]:
        if not image_paths:
            return []
        return [image_paths[i : i + batch_size] for i in range(0, len(image_paths), batch_size)]

    def segment_pdf_by_count(
        self, total_pages: int, max_pages: int = MAX_PDF_PAGES
    ) -> list[tuple[int, int]]:
        if total_pages <= 0:
            return []
        if total_pages <= max_pages:
            return [(0, total_pages)]
        return [(i, min(i + max_pages, total_pages)) for i in range(0, total_pages, max_pages)]

    def compute_video_chunks(
        self, duration_seconds: float, has_audio: bool = True
    ) -> list[tuple[float, float]]:
        max_seconds = MAX_VIDEO_SECONDS_WITH_AUDIO if has_audio else MAX_VIDEO_SECONDS_NO_AUDIO
        if duration_seconds <= 0:
            return []
        if duration_seconds <= max_seconds:
            return [(0.0, duration_seconds)]
        chunks = []
        start = 0.0
        while start < duration_seconds:
            end = min(start + max_seconds, duration_seconds)
            chunks.append((start, end))
            start = end
        return chunks

    def compute_audio_chunks(
        self, duration_seconds: float, max_seconds: int = MAX_AUDIO_SECONDS
    ) -> list[tuple[float, float]]:
        if duration_seconds <= 0:
            return []
        if duration_seconds <= max_seconds:
            return [(0.0, duration_seconds)]
        chunks = []
        start = 0.0
        while start < duration_seconds:
            end = min(start + max_seconds, duration_seconds)
            chunks.append((start, end))
            start = end
        return chunks

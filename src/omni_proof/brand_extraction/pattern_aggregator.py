"""Pattern aggregation for brand identity extraction."""

import logging
import uuid
from collections import Counter

import numpy as np

from omni_proof.brand_extraction.models import (
    AssetExtraction,
    BrandRule,
    BrandVisualStyle,
    BrandVoice,
)

logger = logging.getLogger(__name__)


class PatternAggregator:
    """Aggregates brand patterns from multiple asset extractions."""

    def aggregate(
        self, extractions: list[AssetExtraction]
    ) -> tuple[list[BrandRule], BrandVoice, BrandVisualStyle, list[float], dict[str, float]]:
        """
        Aggregate brand patterns from asset extractions.

        Args:
            extractions: List of AssetExtraction objects

        Returns:
            Tuple of (rules, voice, visual_style, visual_fingerprint, confidence_scores)

        Raises:
            ValueError: If extractions list is empty
        """
        if not extractions:
            raise ValueError("Cannot aggregate empty extractions list")

        # Aggregate colors
        color_counter = Counter()
        for extraction in extractions:
            color_counter.update(extraction.structured_metadata.colors.hex_codes)

        # Top 5 dominant colors
        dominant_colors = [color for color, _ in color_counter.most_common(5)]

        # Color confidence: fraction of assets containing top 3 colors
        top_3_colors = set([color for color, _ in color_counter.most_common(3)])
        assets_with_top_colors = sum(
            1
            for extraction in extractions
            if any(
                color in top_3_colors for color in extraction.structured_metadata.colors.hex_codes
            )
        )
        color_confidence = assets_with_top_colors / len(extractions) if top_3_colors else 0.0

        # Aggregate typography
        font_style_counter = Counter()
        font_name_counter = Counter()
        for extraction in extractions:
            font_style_counter.update(extraction.structured_metadata.typography.font_styles)
            font_name_counter.update(extraction.structured_metadata.typography.font_names)

        typography_styles = [style for style, _ in font_style_counter.most_common(5)]
        typography_fonts = [font for font, _ in font_name_counter.most_common(5)]

        # Typography confidence: fraction of assets with most common style
        most_common_style = font_style_counter.most_common(1)[0][0] if font_style_counter else ""
        assets_with_common_style = sum(
            1
            for extraction in extractions
            if most_common_style in extraction.structured_metadata.typography.font_styles
        )
        typography_confidence = (
            assets_with_common_style / len(extractions) if most_common_style else 0.0
        )

        # Create typography rule
        typography_rule = BrandRule(
            rule_id=str(uuid.uuid4()),
            section_type="typography",
            description=f"Dominant typography styles: {', '.join(typography_styles[:3])}",
            approved_fonts=typography_fonts,
        )

        # Create color rule
        color_rule = BrandRule(
            rule_id=str(uuid.uuid4()),
            section_type="color_palette",
            description=f"Dominant color palette with {len(dominant_colors)} primary colors",
            hex_codes=dominant_colors,
        )

        # Aggregate voice
        formality_counter = Counter(
            extraction.structured_metadata.tone.formality for extraction in extractions
        )
        register_counter = Counter(
            extraction.structured_metadata.tone.emotional_register for extraction in extractions
        )

        dominant_formality = formality_counter.most_common(1)[0][0]
        dominant_register = register_counter.most_common(1)[0][0]

        # Union of vocabulary themes (deduplicated, max 20)
        all_themes = []
        for extraction in extractions:
            all_themes.extend(extraction.structured_metadata.tone.vocabulary_themes)
        vocab_themes = list(dict.fromkeys(all_themes))[
            :20
        ]  # Deduplicate preserving order, limit to 20

        # Voice confidence: fraction matching dominant formality + register
        matching_voice = sum(
            1
            for extraction in extractions
            if (
                extraction.structured_metadata.tone.formality == dominant_formality
                and extraction.structured_metadata.tone.emotional_register == dominant_register
            )
        )
        voice_confidence = matching_voice / len(extractions)

        # Collect sentence styles for mode
        sentence_styles = [
            extraction.structured_metadata.tone.key_phrases[0]
            if extraction.structured_metadata.tone.key_phrases
            else "conversational"
            for extraction in extractions
        ]
        sentence_style_counter = Counter(sentence_styles)
        dominant_sentence_style = (
            sentence_style_counter.most_common(1)[0][0]
            if sentence_style_counter
            else "conversational"
        )

        brand_voice = BrandVoice(
            formality=dominant_formality,
            emotional_register=dominant_register,
            vocabulary_themes=vocab_themes,
            sentence_style=dominant_sentence_style,
            confidence=voice_confidence,
        )

        # Aggregate visual patterns
        layout_counter = Counter(
            extraction.structured_metadata.visual.layout_pattern for extraction in extractions
        )
        motion_counter = Counter(
            extraction.structured_metadata.visual.motion_intensity for extraction in extractions
        )

        layout_patterns = [layout for layout, _ in layout_counter.most_common(3)]
        dominant_motion = motion_counter.most_common(1)[0][0]

        brand_visual_style = BrandVisualStyle(
            dominant_colors=dominant_colors,
            color_consistency=color_confidence,
            typography_styles=typography_styles,
            layout_patterns=layout_patterns,
            motion_style=dominant_motion,
            confidence=color_confidence,  # Use color confidence as overall visual confidence
        )

        # Visual fingerprint: mean of all embeddings, L2-normalized
        embeddings_array = np.array([extraction.embedding for extraction in extractions])
        centroid = np.mean(embeddings_array, axis=0)
        norm = np.linalg.norm(centroid)
        visual_fingerprint = (centroid / norm).tolist() if norm > 0 else centroid.tolist()

        # Outlier detection: log warning for assets with cosine similarity < 0.7
        for extraction in extractions:
            embedding_vec = np.array(extraction.embedding)
            embedding_norm = np.linalg.norm(embedding_vec)
            if embedding_norm > 0:
                cosine_sim = np.dot(embedding_vec, centroid) / (embedding_norm * norm)
                if cosine_sim < 0.7:
                    logger.warning(
                        f"Asset {extraction.asset_path} is an outlier (cosine similarity: {cosine_sim:.3f})"
                    )

        # Confidence scores
        confidence_scores = {
            "color": color_confidence,
            "typography": typography_confidence,
            "voice": voice_confidence,
            "visual": color_confidence,  # Use color confidence for visual
        }

        rules = [color_rule, typography_rule]

        return rules, brand_voice, brand_visual_style, visual_fingerprint, confidence_scores

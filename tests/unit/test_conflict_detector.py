"""Unit tests for ConflictDetector."""

from datetime import datetime, timezone

import pytest

from omni_proof.brand_extraction.conflict_detector import ConflictDetector
from omni_proof.brand_extraction.models import (
    BrandProfile,
    BrandRule,
    BrandVisualStyle,
    BrandVoice,
)


def _make_profile(**overrides) -> BrandProfile:
    """Helper to create a test BrandProfile with defaults."""
    defaults = {
        "profile_id": "test_profile",
        "brand_name": "TestBrand",
        "rules": [],
        "voice": BrandVoice(
            formality="formal",
            emotional_register="professional",
            vocabulary_themes=["innovation", "quality"],
            sentence_style="complex",
            confidence=0.9,
        ),
        "visual_style": BrandVisualStyle(
            dominant_colors=["#FF0000", "#0000FF"],
            color_consistency=0.85,
            typography_styles=["Helvetica", "Arial"],
            layout_patterns=["grid"],
            motion_style="dynamic",
            confidence=0.9,
        ),
        "visual_fingerprint": [0.1] * 3072,
        "source_assets": ["asset1.png"],
        "confidence_scores": {},
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return BrandProfile(**defaults)


def _make_voice(**overrides) -> BrandVoice:
    """Helper to create a test BrandVoice with defaults."""
    defaults = {
        "formality": "formal",
        "emotional_register": "professional",
        "vocabulary_themes": ["innovation", "quality"],
        "sentence_style": "complex",
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return BrandVoice(**defaults)


def _make_visual_style(**overrides) -> BrandVisualStyle:
    """Helper to create a test BrandVisualStyle with defaults."""
    defaults = {
        "dominant_colors": ["#FF0000", "#0000FF"],
        "color_consistency": 0.85,
        "typography_styles": ["Helvetica", "Arial"],
        "layout_patterns": ["grid"],
        "motion_style": "dynamic",
        "confidence": 0.9,
    }
    defaults.update(overrides)
    return BrandVisualStyle(**defaults)


class TestConflictDetector:
    """Test suite for ConflictDetector."""

    def test_no_conflicts_when_consistent(self):
        """Test that no conflicts are detected when patterns are consistent."""
        detector = ConflictDetector()
        existing_profile = _make_profile()

        # New data matches existing exactly
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style()

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        assert len(conflicts) == 0, "Expected no conflicts when patterns are consistent"

    def test_major_color_conflict(self):
        """Test that completely different color palette creates major conflict."""
        detector = ConflictDetector()
        existing_profile = _make_profile()

        # Completely different color palette
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style(
            dominant_colors=["#00FF00", "#FFFF00"]  # Green and Yellow instead of Red and Blue
        )

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have exactly 1 major color conflict
        color_conflicts = [c for c in conflicts if c.dimension == "color_palette"]
        assert len(color_conflicts) == 1, "Expected exactly one color conflict"
        assert color_conflicts[0].severity == "major", (
            "Expected major severity for completely different palette"
        )
        assert (
            "#FF0000" in color_conflicts[0].existing_value
            or "#0000FF" in color_conflicts[0].existing_value
        )
        assert (
            "#00FF00" in color_conflicts[0].new_value or "#FFFF00" in color_conflicts[0].new_value
        )

    def test_major_voice_conflict(self):
        """Test that formal->casual voice change creates major conflict."""
        detector = ConflictDetector()
        existing_profile = _make_profile(
            voice=_make_voice(formality="formal", emotional_register="professional")
        )

        # Change to casual voice
        new_rules = []
        new_voice = _make_voice(formality="casual", emotional_register="friendly")
        new_visual_style = _make_visual_style()

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have 2 major voice conflicts (formality + emotional_register)
        voice_conflicts = [
            c for c in conflicts if c.dimension in ["voice_formality", "voice_emotional_register"]
        ]
        assert len(voice_conflicts) == 2, "Expected two voice conflicts"

        formality_conflicts = [c for c in voice_conflicts if c.dimension == "voice_formality"]
        assert len(formality_conflicts) == 1
        assert formality_conflicts[0].severity == "major"
        assert formality_conflicts[0].existing_value == "formal"
        assert formality_conflicts[0].new_value == "casual"

        register_conflicts = [
            c for c in voice_conflicts if c.dimension == "voice_emotional_register"
        ]
        assert len(register_conflicts) == 1
        assert register_conflicts[0].severity == "major"
        assert register_conflicts[0].existing_value == "professional"
        assert register_conflicts[0].new_value == "friendly"

    def test_minor_typography_addition(self):
        """Test that adding new font style creates major conflict (per spec)."""
        detector = ConflictDetector()
        existing_profile = _make_profile()

        # Add new typography style
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style(
            typography_styles=["Helvetica", "Arial", "Times New Roman"]  # Added Times
        )

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have exactly 1 major typography conflict
        typo_conflicts = [c for c in conflicts if c.dimension == "typography"]
        assert len(typo_conflicts) == 1, "Expected exactly one typography conflict"
        assert typo_conflicts[0].severity == "major", "Expected major severity for new font style"
        assert "Times New Roman" in typo_conflicts[0].new_value

    def test_minor_color_conflict_partial_overlap(self):
        """Test that partial color overlap creates minor conflict."""
        detector = ConflictDetector()
        existing_profile = _make_profile()

        # Partial overlap: keep red, change blue to green
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style(
            dominant_colors=["#FF0000", "#00FF00"]  # Red (kept) and Green (new)
        )

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have exactly 1 minor color conflict
        color_conflicts = [c for c in conflicts if c.dimension == "color_palette"]
        assert len(color_conflicts) == 1, "Expected exactly one color conflict"
        assert color_conflicts[0].severity == "minor", "Expected minor severity for partial overlap"

    def test_motion_style_change_minor_conflict(self):
        """Test that motion style change creates minor conflict (when not static)."""
        detector = ConflictDetector()
        existing_profile = _make_profile(visual_style=_make_visual_style(motion_style="dynamic"))

        # Change motion style
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style(motion_style="smooth")

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have exactly 1 minor motion conflict
        motion_conflicts = [c for c in conflicts if c.dimension == "motion_style"]
        assert len(motion_conflicts) == 1, "Expected exactly one motion conflict"
        assert motion_conflicts[0].severity == "minor", "Expected minor severity for motion change"
        assert motion_conflicts[0].existing_value == "dynamic"
        assert motion_conflicts[0].new_value == "smooth"

    def test_motion_style_no_conflict_when_static(self):
        """Test that motion style change from static doesn't create conflict."""
        detector = ConflictDetector()
        existing_profile = _make_profile(visual_style=_make_visual_style(motion_style="static"))

        # Change motion style from static
        new_rules = []
        new_voice = _make_voice()
        new_visual_style = _make_visual_style(motion_style="dynamic")

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have no motion conflicts
        motion_conflicts = [c for c in conflicts if c.dimension == "motion_style"]
        assert len(motion_conflicts) == 0, "Expected no motion conflict when existing is static"

    def test_multiple_conflicts_combined(self):
        """Test detection of multiple different conflict types."""
        detector = ConflictDetector()
        existing_profile = _make_profile()

        # Multiple changes
        new_rules = []
        new_voice = _make_voice(
            formality="casual"  # Voice conflict
        )
        new_visual_style = _make_visual_style(
            dominant_colors=["#00FF00", "#FFFF00"],  # Color conflict
            typography_styles=["Helvetica", "Arial", "Comic Sans"],  # Typography conflict
            motion_style="smooth",  # Motion conflict
        )

        conflicts = detector.detect(existing_profile, new_rules, new_voice, new_visual_style)

        # Should have multiple conflicts
        assert len(conflicts) >= 4, "Expected at least 4 conflicts"

        conflict_dims = {c.dimension for c in conflicts}
        assert "color_palette" in conflict_dims
        assert "typography" in conflict_dims
        assert "voice_formality" in conflict_dims
        assert "motion_style" in conflict_dims

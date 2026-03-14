"""Conflict detection for brand identity updates."""

from omni_proof.brand_extraction.models import (
    BrandConflict,
    BrandProfile,
    BrandRule,
    BrandVisualStyle,
    BrandVoice,
)


class ConflictDetector:
    """Detects conflicts between existing brand profile and new extracted patterns."""

    def detect(
        self,
        existing_profile: BrandProfile,
        new_rules: list[BrandRule],
        new_voice: BrandVoice,
        new_visual_style: BrandVisualStyle,
    ) -> list[BrandConflict]:
        """
        Detect conflicts between existing profile and new data.

        Args:
            existing_profile: Current brand profile
            new_rules: New brand rules (not used for conflict detection currently)
            new_voice: Newly extracted voice patterns
            new_visual_style: Newly extracted visual style

        Returns:
            List of detected conflicts with severity and details
        """
        conflicts = []

        # Color conflicts
        conflicts.extend(
            self._detect_color_conflicts(existing_profile.visual_style, new_visual_style)
        )

        # Typography conflicts
        conflicts.extend(
            self._detect_typography_conflicts(existing_profile.visual_style, new_visual_style)
        )

        # Voice conflicts
        conflicts.extend(self._detect_voice_conflicts(existing_profile.voice, new_voice))

        # Visual motion conflicts
        conflicts.extend(
            self._detect_motion_conflicts(existing_profile.visual_style, new_visual_style)
        )

        return conflicts

    def _detect_color_conflicts(
        self, existing: BrandVisualStyle, new: BrandVisualStyle
    ) -> list[BrandConflict]:
        """Detect color palette conflicts."""
        conflicts = []

        existing_colors = set(existing.dominant_colors)
        new_colors = set(new.dominant_colors)

        # Calculate overlap
        overlap = existing_colors & new_colors
        new_only = new_colors - existing_colors

        if not overlap and new_colors:
            # No overlap = major conflict
            conflicts.append(
                BrandConflict(
                    dimension="color_palette",
                    existing_value=", ".join(sorted(existing.dominant_colors)),
                    new_value=", ".join(sorted(new.dominant_colors)),
                    source_assets=[],
                    severity="major",
                )
            )
        elif new_only:
            # Some overlap but new colors added = minor conflict
            conflicts.append(
                BrandConflict(
                    dimension="color_palette",
                    existing_value=", ".join(sorted(existing.dominant_colors)),
                    new_value=", ".join(sorted(new.dominant_colors)),
                    source_assets=[],
                    severity="minor",
                )
            )

        return conflicts

    def _detect_typography_conflicts(
        self, existing: BrandVisualStyle, new: BrandVisualStyle
    ) -> list[BrandConflict]:
        """Detect typography conflicts."""
        conflicts = []

        existing_typo = set(existing.typography_styles)
        new_typo = set(new.typography_styles)

        # New styles not in existing = major conflict
        new_only = new_typo - existing_typo
        if new_only:
            conflicts.append(
                BrandConflict(
                    dimension="typography",
                    existing_value=", ".join(sorted(existing.typography_styles)),
                    new_value=", ".join(sorted(new.typography_styles)),
                    source_assets=[],
                    severity="major",
                )
            )

        return conflicts

    def _detect_voice_conflicts(self, existing: BrandVoice, new: BrandVoice) -> list[BrandConflict]:
        """Detect brand voice conflicts."""
        conflicts = []

        # Formality changed = major
        if existing.formality != new.formality:
            conflicts.append(
                BrandConflict(
                    dimension="voice_formality",
                    existing_value=existing.formality,
                    new_value=new.formality,
                    source_assets=[],
                    severity="major",
                )
            )

        # Emotional register changed = major
        if existing.emotional_register != new.emotional_register:
            conflicts.append(
                BrandConflict(
                    dimension="voice_emotional_register",
                    existing_value=existing.emotional_register,
                    new_value=new.emotional_register,
                    source_assets=[],
                    severity="major",
                )
            )

        return conflicts

    def _detect_motion_conflicts(
        self, existing: BrandVisualStyle, new: BrandVisualStyle
    ) -> list[BrandConflict]:
        """Detect motion style conflicts."""
        conflicts = []

        # Motion style changed (and existing isn't "static") = minor
        if existing.motion_style != new.motion_style and existing.motion_style != "static":
            conflicts.append(
                BrandConflict(
                    dimension="motion_style",
                    existing_value=existing.motion_style,
                    new_value=new.motion_style,
                    source_assets=[],
                    severity="minor",
                )
            )

        return conflicts

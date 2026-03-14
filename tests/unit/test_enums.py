"""Tests for creative metadata enum types."""

from omni_proof.ingestion.enums import (
    AudioGenre,
    BackgroundSetting,
    CTAType,
    EmotionalTone,
    TypographyStyle,
    VoiceoverDemographic,
)


def test_background_setting_values():
    assert BackgroundSetting.INDOOR == "indoor"
    assert BackgroundSetting.OUTDOOR == "outdoor"
    assert BackgroundSetting.ABSTRACT == "abstract"
    assert BackgroundSetting.STUDIO == "studio"


def test_cta_type_values():
    assert CTAType.URGENCY == "urgency"
    assert CTAType.PASSIVE == "passive"
    assert CTAType.INQUISITIVE == "inquisitive"
    assert CTAType.IMPERATIVE == "imperative"


def test_emotional_tone_values():
    assert EmotionalTone.AUTHORITATIVE == "authoritative"
    assert EmotionalTone.CONVERSATIONAL == "conversational"
    assert EmotionalTone.ENERGETIC == "energetic"
    assert EmotionalTone.CALM == "calm"


def test_audio_genre_values():
    assert AudioGenre.POP == "pop"
    assert AudioGenre.ELECTRONIC == "electronic"
    assert AudioGenre.AMBIENT == "ambient"
    assert AudioGenre.CLASSICAL == "classical"
    assert AudioGenre.NONE == "none"


def test_typography_style_values():
    assert TypographyStyle.SERIF == "serif"
    assert TypographyStyle.SANS_SERIF == "sans_serif"
    assert TypographyStyle.DISPLAY == "display"
    assert TypographyStyle.HANDWRITTEN == "handwritten"


def test_voiceover_demographic_values():
    assert VoiceoverDemographic.MALE_YOUNG == "male_young"
    assert VoiceoverDemographic.FEMALE_YOUNG == "female_young"
    assert VoiceoverDemographic.MALE_MATURE == "male_mature"
    assert VoiceoverDemographic.FEMALE_MATURE == "female_mature"
    assert VoiceoverDemographic.NONE == "none"

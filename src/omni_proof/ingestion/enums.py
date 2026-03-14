"""Enum types for creative metadata extraction taxonomy."""

from enum import Enum


class BackgroundSetting(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ABSTRACT = "abstract"
    STUDIO = "studio"
    URBAN = "urban"
    NATURE = "nature"


class CTAType(str, Enum):
    URGENCY = "urgency"
    PASSIVE = "passive"
    INQUISITIVE = "inquisitive"
    IMPERATIVE = "imperative"


class EmotionalTone(str, Enum):
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    ENERGETIC = "energetic"
    CALM = "calm"
    HUMOROUS = "humorous"
    INSPIRATIONAL = "inspirational"


class AudioGenre(str, Enum):
    POP = "pop"
    ELECTRONIC = "electronic"
    AMBIENT = "ambient"
    CLASSICAL = "classical"
    HIP_HOP = "hip_hop"
    ROCK = "rock"
    NONE = "none"


class TypographyStyle(str, Enum):
    SERIF = "serif"
    SANS_SERIF = "sans_serif"
    DISPLAY = "display"
    HANDWRITTEN = "handwritten"
    MONOSPACE = "monospace"


class VoiceoverDemographic(str, Enum):
    MALE_YOUNG = "male_young"
    FEMALE_YOUNG = "female_young"
    MALE_MATURE = "male_mature"
    FEMALE_MATURE = "female_mature"
    NONE = "none"

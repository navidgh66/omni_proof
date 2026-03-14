"""SQLAlchemy ORM models mirroring Pydantic schemas for relational storage."""

from sqlalchemy import JSON, BigInteger, Date, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CreativeMetadataRecord(Base):
    __tablename__ = "creative_metadata"

    asset_id: Mapped[str] = mapped_column(String, primary_key=True)
    campaign_id: Mapped[str] = mapped_column(String, nullable=False)
    objects_detected: Mapped[list] = mapped_column(JSON, default=list)
    logo_screen_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    background_setting: Mapped[str] = mapped_column(String, default="")
    dominant_colors: Mapped[list] = mapped_column(JSON, default=list)
    contrast_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    scene_transitions: Mapped[int] = mapped_column(Integer, default=0)
    time_to_first_logo: Mapped[float] = mapped_column(Float, default=0.0)
    product_exposure_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    motion_intensity: Mapped[float] = mapped_column(Float, default=0.0)
    text_density: Mapped[float] = mapped_column(Float, default=0.0)
    cta_type: Mapped[str] = mapped_column(String, default="")
    promotional_text: Mapped[str] = mapped_column(Text, default="")
    typography_style: Mapped[str] = mapped_column(String, default="")
    audio_genre: Mapped[str] = mapped_column(String, default="")
    voiceover_tone: Mapped[str] = mapped_column(String, default="")
    music_tempo_bpm: Mapped[int] = mapped_column(Integer, default=0)
    platform: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[str] = mapped_column(DateTime, nullable=True)


class PerformanceRecord(Base):
    __tablename__ = "performance_records"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    asset_id: Mapped[str] = mapped_column(String, nullable=False)
    impressions: Mapped[int] = mapped_column(BigInteger, default=0)
    clicks: Mapped[int] = mapped_column(BigInteger, default=0)
    conversions: Mapped[int] = mapped_column(Integer, default=0)
    roas: Mapped[float] = mapped_column(Float, default=0.0)
    ctr: Mapped[float] = mapped_column(Float, default=0.0)
    audience_segment: Mapped[str] = mapped_column(String, default="")
    date: Mapped[str] = mapped_column(Date, nullable=True)
    platform: Mapped[str] = mapped_column(String, default="")


class CampaignRecord(Base):
    __tablename__ = "campaigns"

    campaign_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    start_date: Mapped[str] = mapped_column(Date, nullable=True)
    end_date: Mapped[str] = mapped_column(Date, nullable=True)
    budget: Mapped[float] = mapped_column(Float, default=0.0)
    target_demographics: Mapped[dict] = mapped_column(JSON, default=dict)

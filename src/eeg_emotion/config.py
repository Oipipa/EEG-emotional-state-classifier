from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import yaml


def _default_bands() -> Dict[str, Tuple[float, float]]:
    return {
        "Delta": (1.0, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 45.0),
    }


@dataclass
class PathsConfig:
    """Filesystem layout used by the pipeline."""

    raw: Path = Path("data/raw")
    processed: Path = Path("data/processed")
    models: Path = Path("models")
    logs: Path = Path("logs")


@dataclass
class ChannelConfig:
    """EEG, marker, quality and battery channel definitions."""

    eeg_channels: Sequence[str] = ("AF3", "T7", "Pz", "T8", "AF4")
    marker_channels: Sequence[str] = ("MarkerType", "MarkerValueInt", "MARKER_HARDWARE")
    cq_channels: Sequence[str] = (
        "CQ_AF3",
        "CQ_T7",
        "CQ_Pz",
        "CQ_T8",
        "CQ_AF4",
        "CQ_OVERALL",
    )
    eq_channels: Sequence[str] = (
        "EQ_AF3",
        "EQ_T7",
        "EQ_Pz",
        "EQ_T8",
        "EQ_AF4",
        "EQ_OVERALL",
    )
    battery_channels: Sequence[str] = ("BATTERY", "BATTERY_PERCENT")

    @property
    def quality_channels(self) -> List[str]:
        return list(self.cq_channels) + list(self.eq_channels)


@dataclass
class FeatureConfig:
    """Parameters for feature engineering."""

    bands: Dict[str, Tuple[float, float]] = field(default_factory=_default_bands)
    sampling_rate: int = 128
    min_duration_sec: int = 10
    pca_components: int = 20
    random_seed: int = 42
    winsorize: bool = True


@dataclass
class ModelConfig:
    """Training hyperparameters for the neural network baseline."""

    hidden_dims: Sequence[int] = (256, 128, 64, 128, 256)
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 100
    test_frac: float = 0.2


@dataclass
class PipelineConfig:
    """Top-level configuration bundle."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)

    @staticmethod
    def from_yaml(path: Path | str | None = None) -> "PipelineConfig":
        if path is None:
            return PipelineConfig()
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f) or {}

        def merge_dataclass(dc_cls, values):
            return dc_cls(**values) if isinstance(values, dict) else dc_cls()

        paths = merge_dataclass(PathsConfig, data.get("paths", {}))
        channels = merge_dataclass(ChannelConfig, data.get("channels", {}))
        features = merge_dataclass(FeatureConfig, data.get("features", {}))
        model = merge_dataclass(ModelConfig, data.get("model", {}))
        return PipelineConfig(paths=paths, channels=channels, features=features, model=model)

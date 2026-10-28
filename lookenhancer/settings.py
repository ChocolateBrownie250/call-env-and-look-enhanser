from dataclasses import dataclass

RESOLUTIONS = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "540p": (960, 540),
    "480p": (854, 480),
    "360p": (640, 360),
}

PRESETS = {
    "Natural": dict(smooth=0.25, tone=0.15, eyes=0.10, exposure=0.5, warmth=0.0),
    "Studio": dict(smooth=0.45, tone=0.30, eyes=0.25, exposure=0.6, warmth=0.1),
    "Glam": dict(smooth=0.75, tone=0.50, eyes=0.40, exposure=0.7, warmth=0.2),
}


@dataclass
class Settings:
    enabled: bool = True
    compare: bool = False
    smooth: float = 0.45
    tone: float = 0.30
    eyes: float = 0.25
    exposure: float = 0.6
    warmth: float = 0.1

    resolution: str = "720p"
    fps: int = 30

    softbox_on: bool = True
    softbox_color: tuple = (255, 244, 229)
    softbox_level: float = 0.6
    softbox_width: float = 0.07
    softbox_edges: str = "tlr"
    softbox_auto: bool = True
    softbox_target: float = 0.5

    face_luma: float = 0.0
    face_found: bool = False
    ms: float = 0.0
    status: str = "starting"
    running: bool = True

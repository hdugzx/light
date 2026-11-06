"""Capture a still frame from the computer's webcam using OpenCV.

Also exposes brightness features that give a model-independent signal for
"is a bright light (e.g. a flashlight) visible in the frame?".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import CameraConfig


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    path: Path
    width: int
    height: int
    # Brightness features in 0..255 / 0..1 ranges.
    mean_brightness: float
    max_region_brightness: float
    bright_pixel_fraction: float  # fraction of pixels above a high threshold

    def describe(self) -> str:
        return (
            f"image_size={self.width}x{self.height}, "
            f"mean_brightness={self.mean_brightness:.1f}/255, "
            f"brightest_region={self.max_region_brightness:.1f}/255, "
            f"bright_pixel_fraction={self.bright_pixel_fraction:.3f}"
        )


def _extract_features(gray) -> tuple[float, float, float]:
    import cv2  # local import: only needed when a camera is used
    import numpy as np

    mean_brightness = float(gray.mean())
    # Blur first so a single hot pixel doesn't dominate the "brightest region".
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=9)
    max_region_brightness = float(blurred.max())
    bright_pixel_fraction = float((gray > 230).mean())
    return mean_brightness, max_region_brightness, bright_pixel_fraction


def capture(cfg: CameraConfig) -> Frame:
    """Grab one frame from the webcam and persist it as a JPEG."""
    import cv2

    cap = cv2.VideoCapture(cfg.index)
    if not cap.isOpened():
        raise CameraError(
            f"Could not open camera index {cfg.index}. "
            "Check the device is connected and not in use, or set CAMERA_INDEX."
        )
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        # Warm up: many webcams need a few frames to auto-expose.
        frame = None
        for _ in range(max(1, cfg.warmup_frames)):
            ok, frame = cap.read()
            if not ok:
                frame = None
            time.sleep(0.05)
        if frame is None:
            raise CameraError("Camera opened but returned no frame.")
    finally:
        cap.release()

    return _persist(cfg, frame)


def load_image(cfg: CameraConfig, image_path: str | Path) -> Frame:
    """Use an existing image instead of the live camera (handy for testing)."""
    import cv2

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise CameraError(f"Could not read image file: {image_path}")
    return _persist(cfg, frame, source=Path(image_path))


def _persist(cfg: CameraConfig, frame, source: Path | None = None) -> Frame:
    import cv2

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = cfg.save_dir / f"capture_{stamp}.jpg"
    cv2.imwrite(str(out_path), frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_b, max_b, frac = _extract_features(gray)
    h, w = gray.shape[:2]
    return Frame(
        path=out_path,
        width=int(w),
        height=int(h),
        mean_brightness=mean_b,
        max_region_brightness=max_b,
        bright_pixel_fraction=frac,
    )

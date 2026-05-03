"""In-memory image registry and bbox overlay helpers."""

from __future__ import annotations

import uuid
from typing import Any

from PIL import Image, ImageDraw, ImageFont


class CanvasRegistry:
    """Maps JSON-friendly image ids to PIL images."""

    def __init__(self, recorder: Any | None = None) -> None:
        self._images: dict[str, Image.Image] = {}
        self._recorder = recorder

    def register(self, image: Image.Image, artifact_name: str | None = None) -> str:
        image_id = str(uuid.uuid4())
        self._images[image_id] = image.copy()
        if self._recorder is not None:
            self._recorder.save_image(image_id, image, artifact_name=artifact_name)
        return image_id

    def get(self, image_id: str) -> Image.Image:
        if image_id not in self._images:
            raise KeyError(f"Unknown image_id: {image_id}")
        return self._images[image_id].copy()


def draw_bbox_overlay(image: Image.Image, layout: dict[str, Any]) -> Image.Image:
    """Render each planned object bbox and label over a copy of the image."""

    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for index, obj in enumerate(layout.get("objects", []), start=1):
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, width, height = [int(value) for value in bbox]
        x2 = x + width
        y2 = y + height
        label = f"{index}. {obj.get('name', 'object')}"
        draw.rectangle((x, y, x2, y2), outline="red", width=4)
        text_box = draw.textbbox((x, y), label, font=font)
        pad = 3
        draw.rectangle(
            (
                text_box[0] - pad,
                text_box[1] - pad,
                text_box[2] + pad,
                text_box[3] + pad,
            ),
            fill="white",
        )
        draw.text((x, y), label, fill="red", font=font)

    return overlay

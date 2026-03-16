from __future__ import annotations

from pathlib import Path

import pygame


def _pil_font_path() -> str | None:
    # repo root: .../game/renderer/text.py -> parents[2] == project root
    root = Path(__file__).resolve().parents[2]
    p = root / "JetBrainsMonoNL-Regular.ttf"
    if p.exists():
        return str(p)
    p = Path.cwd() / "JetBrainsMonoNL-Regular.ttf"
    return str(p) if p.exists() else None


class PILTextRenderer:
    """Рендер текста через PIL → pygame.Surface (без pygame.font)."""

    def __init__(self) -> None:
        self._pil_font = None
        self._label_cache: dict[str, pygame.Surface] = {}

    def get_cached_label(self, label: str) -> pygame.Surface | None:
        return self._label_cache.get(label)

    def set_cached_label(self, label: str, surface: pygame.Surface | None) -> None:
        if surface is not None:
            self._label_cache[label] = surface

    def _get_pil_font(self):
        if self._pil_font is not None:
            return self._pil_font
        try:
            from PIL import ImageFont

            path = _pil_font_path()
            if path:
                self._pil_font = ImageFont.truetype(path, 16)
            else:
                self._pil_font = ImageFont.load_default()
            return self._pil_font
        except Exception:
            return None

    def render(self, text: str, color: tuple[int, int, int] = (220, 220, 220)) -> pygame.Surface | None:
        """
        Рендер текста через PIL → pygame.Surface (без pygame.font).
        Отступы сверху/снизу — чтобы не обрезало.
        """
        font = self._get_pil_font()
        if font is None:
            return None
        try:
            from PIL import Image, ImageDraw

            bbox = font.getbbox(text)
            pad_x = 6
            pad_top = 4
            pad_bottom = 6
            w = max(1, bbox[2] - bbox[0] + 2 * pad_x)
            h = max(1, bbox[3] - bbox[1] + pad_top + pad_bottom)
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((pad_x, pad_top), text, font=font, fill=(*color, 255))
            data = img.tobytes()
            surf = pygame.image.frombytes(data, (w, h), "RGBA")
            return surf
        except Exception:
            return None


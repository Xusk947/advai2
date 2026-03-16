from __future__ import annotations

import pygame

from .constants import (
    ASSETS_DIR,
    EYE_PLATFORM_IMAGE,
    EYE_PUPIL_IMAGE,
    GHOST_BODY_FRAMES,
    PACMAN_FRAMES,
)


def _load_image_any(path) -> pygame.Surface:
    """
    Грузит изображение в pygame.Surface.

    В некоторых сборках pygame `pygame.image.get_extended()` == False и PNG не поддерживается.
    Тогда пробуем через PIL и конвертируем в Surface.
    """
    try:
        return pygame.image.load(str(path)).convert_alpha()
    except Exception:
        # Fallback: PIL -> RGBA bytes -> pygame Surface
        from PIL import Image

        img = Image.open(path).convert("RGBA")
        data = img.tobytes()
        surf = pygame.image.frombytes(data, img.size, "RGBA")
        # convert_alpha works only after display init; if not initialized, ignore.
        try:
            return surf.convert_alpha()
        except Exception:
            return surf


def load_ghost_sprites(tile_size: int):
    """
    Загружает спрайты призрака, масштабирует под tile_size.
    Возвращает (body_frames, eye_platform, eye_pupil) или (None, None, None).
    """
    out = []
    for name in GHOST_BODY_FRAMES:
        p = ASSETS_DIR / name
        if not p.exists():
            return (None, None, None)
        try:
            s = _load_image_any(p)
            if s.get_size()[0] != tile_size or s.get_size()[1] != tile_size:
                s = pygame.transform.scale(s, (tile_size, tile_size))  # scale = чёткий пиксель-арт, без размытия
            out.append(s)
        except (pygame.error, OSError):
            return (None, None, None)

    ep_path = ASSETS_DIR / EYE_PLATFORM_IMAGE
    eye_pl = None
    if ep_path.exists():
        try:
            eye_pl = _load_image_any(ep_path)
            # Платформа глаза (белок): ближе к оригиналу Pac-Man — ~35–40% тайла
            ew = max(2, tile_size * 9 // 25)
            eh = max(2, tile_size * 9 // 25)
            eye_pl = pygame.transform.scale(eye_pl, (ew, eh))
        except (pygame.error, OSError):
            pass

    eye_path = ASSETS_DIR / EYE_PUPIL_IMAGE
    eye_pupil = None
    if eye_path.exists():
        try:
            eye_pupil = _load_image_any(eye_path)
            # Зрачок: ~25% тайла, как в оригинале
            es = max(2, tile_size // 4)
            eye_pupil = pygame.transform.scale(eye_pupil, (es, es))
        except (pygame.error, OSError):
            pass

    return (tuple(out), eye_pl, eye_pupil)


def load_pacman_sprites(tile_size: int) -> tuple[pygame.Surface | None, pygame.Surface | None]:
    """
    Загружает кадры Пакмана (pacman_0001.png, pacman_0002.png), масштабирует под tile_size.
    Спрайт по умолчанию смотрит вправо (RIGHT). Возвращает (frame0, frame1) или (None, None).
    """
    out: list[pygame.Surface | None] = []
    for name in PACMAN_FRAMES:
        p = ASSETS_DIR / name
        if not p.exists():
            return (None, None)
        try:
            s = _load_image_any(p)
            if s.get_size()[0] != tile_size or s.get_size()[1] != tile_size:
                s = pygame.transform.scale(s, (tile_size, tile_size))
            out.append(s)
        except (pygame.error, OSError):
            return (None, None)
    return (out[0], out[1])


def tint_white_sprite(surface: pygame.Surface, color: tuple[int, int, int]) -> pygame.Surface:
    """Перекрашивает белые пиксели в color (чёрный контур сохраняется)."""
    out = surface.copy()
    if out.get_bytesize() != 4:
        out = out.convert_alpha()
    tint = pygame.Surface(out.get_size())
    tint.fill((*color, 255))
    out.blit(tint, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    return out


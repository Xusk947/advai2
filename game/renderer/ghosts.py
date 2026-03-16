from __future__ import annotations

import pygame


def draw_ghost_eyes(
    screen: pygame.Surface,
    tile_size: int,
    eye_platform: pygame.Surface,
    eye_pupil: pygame.Surface,
    g_cx: int,
    g_cy: int,
    direction: int,
) -> None:
    """Рисует два глаза (платформа + зрачок), зрачок смещён по direction (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).
    Размеры и расположение подогнаны под пропорции оригинального Pac-Man."""
    ts = tile_size
    pw, ph = eye_platform.get_size()
    # Смещение зрачка по направлению взгляда
    shift = max(2, ts // 10)
    dx = dy = 0
    if direction == 0:
        dy = -shift
    elif direction == 1:
        dy = shift
    elif direction == 2:
        dx = -shift
    else:
        dx = shift
    # Расстояние между центрами глаз ~40% тайла (как в оригинале)
    eye_span = ts * 5 // 20
    left_eye_x = g_cx - eye_span - pw // 2
    right_eye_x = g_cx + eye_span - pw // 2
    # Глаза в верхней части призрака, чуть ниже вершины
    eye_y = g_cy - ts // 4 - ph // 2
    for ex in (left_eye_x, right_eye_x):
        screen.blit(eye_platform, (ex, eye_y))
        pupil_rect = eye_pupil.get_rect(center=(ex + pw // 2 + dx, eye_y + ph // 2 + dy))
        screen.blit(eye_pupil, pupil_rect.topleft)


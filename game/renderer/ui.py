from __future__ import annotations

import pygame

from .constants import (
    DEBUG_REWARDS_BUTTON_VALUE,
    MAP_BUTTON_BASE,
    MAX_STEPS_PRESETS,
    POPUPS_BUTTON_VALUE,
    ROUNDS_BUTTON_BASE,
    SPEED_PRESETS,
    UI_STRIP_HEIGHT,
)


def build_ui_buttons(
    game_width: int,
    game_height: int,
    speed_index: int,
    show_popups: bool = True,
    show_debug_rewards: bool = False,
    max_steps_index: int = 2,
    level_index: int = 0,
) -> list[tuple[pygame.Rect, str, int | None]]:
    margin = 8
    btn_h = 26
    gap = 4
    row_gap = 6
    y0 = game_height + margin
    button_rects: list[tuple[pygame.Rect, str, int | None]] = []

    # Строка 1: Reload, ±, Rwd, скорость
    reload_w = 56
    r = pygame.Rect(margin, y0, reload_w, btn_h)
    button_rects.append((r, "Reload", None))
    x = margin + reload_w + gap

    popups_w = 28
    r = pygame.Rect(x, y0, popups_w, btn_h)
    button_rects.append((r, "±", POPUPS_BUTTON_VALUE))
    x += popups_w + gap

    debug_w = 36
    r = pygame.Rect(x, y0, debug_w, btn_h)
    button_rects.append((r, "Rwd", DEBUG_REWARDS_BUTTON_VALUE))
    x += debug_w + gap

    n = len(SPEED_PRESETS)
    total_w = game_width - 2 * margin - reload_w - gap - popups_w - gap - debug_w - gap
    btn_w = max(44, (total_w - (n - 1) * gap) // n)
    for i, (_, label) in enumerate(SPEED_PRESETS):
        r = pygame.Rect(x, y0, btn_w, btn_h)
        button_rects.append((r, label, i))
        x += btn_w + gap

    # Строка 2: Rounds (64, 128, 256, 284, 512), Map (1, 2)
    y1 = y0 + btn_h + row_gap
    x = margin
    for i, steps in enumerate(MAX_STEPS_PRESETS):
        w = 40
        r = pygame.Rect(x, y1, w, btn_h)
        button_rects.append((r, str(steps), ROUNDS_BUTTON_BASE + i))
        x += w + gap
    x += 8
    for map_num in (1, 2):
        w = 28
        r = pygame.Rect(x, y1, w, btn_h)
        button_rects.append((r, str(map_num), MAP_BUTTON_BASE + map_num))
        x += w + gap

    return button_rects

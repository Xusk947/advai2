from __future__ import annotations

import pygame

from env.ghosts import GhostMode
from env.pacman import Direction

from .assets import tint_white_sprite
from .constants import (
    DEBUG_REWARDS_BUTTON_VALUE,
    MAP_BUTTON_BASE,
    POPUPS_BUTTON_VALUE,
    ROUNDS_BUTTON_BASE,
    UI_STRIP_HEIGHT,
)
from .ghosts import draw_ghost_eyes
from .text import PILTextRenderer


def _draw_ghost_fallback(
    screen: pygame.Surface,
    tile_size: int,
    g_cx: int,
    g_cy: int,
    color: tuple[int, int, int],
    frightened: bool,
    direction: int,
    eyes_only: bool = False,
) -> None:
    """Рисует простого 'призрака' примитивами, если нет спрайтов. eyes_only=True — только глаза (съеден, летит на спавн)."""
    if eyes_only:
        # Только глаза (призрак съеден, возвращается на спавн)
        eye_y = g_cy - tile_size // 5
        eye_dx = tile_size // 6
        eye_r = max(2, tile_size // 8)
        pupil_r = max(1, tile_size // 14)
        shift = max(1, tile_size // 12)
        dx = dy = 0
        if direction == 0:
            dy = -shift
        elif direction == 1:
            dy = shift
        elif direction == 2:
            dx = -shift
        else:
            dx = shift
        outline = (0, 0, 0)
        pupil_color = (40, 120, 255)
        for ex in (g_cx - eye_dx, g_cx + eye_dx):
            pygame.draw.circle(screen, (255, 255, 255), (ex, eye_y), eye_r)
            pygame.draw.circle(screen, outline, (ex, eye_y), eye_r, 1)
            pygame.draw.circle(screen, pupil_color, (ex + dx, eye_y + dy), pupil_r)
        return

    body = pygame.Rect(0, 0, tile_size - 4, tile_size - 4)
    body.center = (g_cx, g_cy)

    r = max(4, (tile_size - 6) // 2)
    head_center = (body.centerx, body.top + r)
    mid_y = body.top + r
    body_rect = pygame.Rect(body.left, mid_y, body.width, body.bottom - mid_y)

    outline = (0, 0, 0)
    pygame.draw.circle(screen, color, head_center, r)
    pygame.draw.rect(screen, color, body_rect)

    bump_r = max(2, body.width // 6)
    bump_y = body.bottom - bump_r
    for bx in (body.left + bump_r, body.centerx, body.right - bump_r):
        pygame.draw.circle(screen, color, (bx, bump_y), bump_r)

    pygame.draw.circle(screen, outline, head_center, r, 2)
    pygame.draw.rect(screen, outline, body_rect, 2)

    if frightened:
        return

    eye_y = body.top + tile_size // 3
    eye_dx = tile_size // 6
    eye_r = max(2, tile_size // 8)
    pupil_r = max(1, tile_size // 14)
    shift = max(1, tile_size // 12)
    dx = dy = 0
    if direction == 0:
        dy = -shift
    elif direction == 1:
        dy = shift
    elif direction == 2:
        dx = -shift
    else:
        dx = shift

    pupil_color = (40, 120, 255)
    for ex in (body.centerx - eye_dx, body.centerx + eye_dx):
        pygame.draw.circle(screen, (255, 255, 255), (ex, eye_y), eye_r)
        pygame.draw.circle(screen, outline, (ex, eye_y), eye_r, 1)
        pygame.draw.circle(screen, pupil_color, (ex + dx, eye_y + dy), pupil_r)


def draw_level(screen: pygame.Surface, level, tile_size: int, colors: dict) -> None:
    for y in range(level.height):
        for x in range(level.width):
            rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
            if level.walls[y][x]:
                pygame.draw.rect(screen, colors["wall"], rect)
            else:
                if level.pills[y][x]:
                    pygame.draw.circle(
                        screen, colors["pill"], rect.center, tile_size // 8
                    )
                if level.power_pills[y][x]:
                    pygame.draw.circle(
                        screen, colors["power_pill"], rect.center, tile_size // 4
                    )


def _direction_to_angle(direction: Direction) -> float:
    """Угол поворота спрайта (спрайт по умолчанию смотрит вправо). pygame: положительный = по часовой."""
    if direction is Direction.RIGHT:
        return 0.0
    if direction is Direction.UP:
        return 90.0  # рот вверх
    if direction is Direction.DOWN:
        return -90.0  # рот вниз
    # LEFT
    return 180.0


def draw_pacman(
    screen: pygame.Surface,
    pacman,
    tile_size: int,
    colors: dict,
    pacman_display: tuple[float, float] | None,
    pacman_frames: tuple[pygame.Surface | None, pygame.Surface | None] = (None, None),
    steps_done: int = 0,
) -> None:
    if pacman_display is not None:
        px, py = pacman_display
    else:
        px, py = pacman.pos.to_tuple()

    p_cx = int(px * tile_size + tile_size // 2)
    p_cy = int(py * tile_size + tile_size // 2)
    p_rect = pygame.Rect(int(px * tile_size), int(py * tile_size), tile_size, tile_size)
    p_rect.center = (p_cx, p_cy)

    frame0, frame1 = pacman_frames
    if frame0 is not None and frame1 is not None:
        # Анимация: переключение кадра чаще (каждые 2 шага)
        frame_idx = (steps_done // 1) % 2
        surf = frame1 if frame_idx else frame0
        angle = _direction_to_angle(pacman.direction)
        rotated = pygame.transform.rotate(surf, angle)
        r_rect = rotated.get_rect(center=(p_cx, p_cy))
        screen.blit(rotated, r_rect.topleft)
        return

    # Fallback: круг и стрелка
    pygame.draw.circle(screen, colors["pacman"], p_rect.center, tile_size // 2 - 2)
    cx, cy = p_cx, p_cy
    ts = tile_size
    d = pacman.direction
    if d is Direction.UP:
        arrow = [(cx, cy - ts // 2), (cx - 6, cy - 8), (cx + 6, cy - 8)]
    elif d is Direction.DOWN:
        arrow = [(cx, cy + ts // 2), (cx - 6, cy + 8), (cx + 6, cy + 8)]
    elif d is Direction.LEFT:
        arrow = [(cx - ts // 2, cy), (cx - 8, cy - 6), (cx - 8, cy + 6)]
    else:
        arrow = [(cx + ts // 2, cy), (cx + 8, cy - 6), (cx + 8, cy + 6)]
    pygame.draw.polygon(screen, (255, 200, 0), arrow)
    pygame.draw.aalines(screen, (200, 150, 0), True, arrow)


def draw_ghosts(
    screen: pygame.Surface,
    ghosts: dict,
    tile_size: int,
    colors: dict,
    ghost_body_frames: tuple[pygame.Surface, ...] | None,
    eye_platform: pygame.Surface | None,
    eye_pupil: pygame.Surface | None,
    tinted_cache: dict[tuple[str, int], pygame.Surface],
    steps_done: int,
    ghost_display: dict[str, tuple[float, float]] | None,
) -> None:
    ghost_frame_idx = (steps_done // 3) % 2 if ghost_body_frames else 0

    for key, ghost in ghosts.items():
        if ghost_display is not None and key in ghost_display:
            gx, gy = ghost_display[key]
        else:
            gx, gy = ghost.pos.to_tuple()

        g_cx = int(gx * tile_size + tile_size // 2)
        g_cy = int(gy * tile_size + tile_size // 2)
        g_rect = pygame.Rect(
            int(gx * tile_size), int(gy * tile_size), tile_size, tile_size
        )
        g_rect.center = (g_cx, g_cy)

        if ghost.mode is GhostMode.FRIGHTENED:
            color = (0, 0, 255)
        else:
            color_key = f"ghost_{key}"
            color = colors.get(color_key, (200, 200, 200))

        eaten = (
            ghost.mode is GhostMode.EATEN
        )  # съеден — только глаза, пока не дойдёт до спавна

        if ghost_body_frames and len(ghost_body_frames) > ghost_frame_idx:
            # Тело рисуем только если не EATEN
            if not eaten:
                cache_key = (
                    "frightened" if ghost.mode is GhostMode.FRIGHTENED else key,
                    ghost_frame_idx,
                )
                if cache_key not in tinted_cache:
                    body = ghost_body_frames[ghost_frame_idx]
                    tinted_cache[cache_key] = tint_white_sprite(body, color)
                body_surf = tinted_cache[cache_key]
                screen.blit(body_surf, g_rect.topleft)

            # Глаза — всегда (и при frightened, и при EATEN — только глаза летят на спавн)
            if eye_platform is not None and eye_pupil is not None:
                draw_ghost_eyes(
                    screen,
                    tile_size,
                    eye_platform,
                    eye_pupil,
                    g_cx,
                    g_cy,
                    ghost.direction,
                )
        else:
            _draw_ghost_fallback(
                screen,
                tile_size,
                g_cx,
                g_cy,
                color,
                frightened=(ghost.mode is GhostMode.FRIGHTENED),
                direction=ghost.direction,
                eyes_only=eaten,
            )


def build_caption(
    steps: int,
    reward: float,
    episode_num: int | None,
    episode_ended: bool,
    max_steps: int | None = None,
    level_num: int | None = None,
) -> str:
    # Короткий тайтл, чтобы не обрезался
    ep = f" ep.{episode_num}" if episode_num is not None else ""
    rwd = f" r {reward:.0f}" if reward != int(reward) else f" r {int(reward)}"
    extra = ""
    if max_steps is not None:
        extra += f" | {max_steps}"
    if level_num is not None:
        extra += f" m{level_num}"
    if episode_ended and episode_num is not None:
        return f"Pacman RL{ep} end s{steps}{rwd}{extra}"
    if episode_num is not None:
        return f"Pacman RL{ep} s{steps}{rwd}{extra}"
    return f"Pacman RL s{steps}{rwd}{extra}"


def draw_reward(screen: pygame.Surface, text: PILTextRenderer, reward: float) -> None:
    reward_surf = text.render(f"Reward: {reward:.1f}", (255, 255, 255))
    if reward_surf is not None:
        screen.blit(reward_surf, (8, 4))


def draw_event_popups(
    screen: pygame.Surface,
    tile_size: int,
    popups: list[tuple[float, float, str, tuple[int, int, int]]],
    text: PILTextRenderer,
) -> None:
    for tile_x, tile_y, msg, color in popups:
        px = int(tile_x * tile_size + tile_size // 2)
        py = int((tile_y + 1) * tile_size) + 6
        surf = text.render(msg, color)
        if surf is not None:
            r = surf.get_rect(center=(px, py))
            screen.blit(surf, r.topleft)


def draw_entity_rewards(
    screen: pygame.Surface,
    tile_size: int,
    pacman_pos: tuple[float, float],
    pacman_reward: float,
    ghost_positions: dict[str, tuple[float, float]],
    ghost_rewards: dict[str, float],
    colors: dict[str, tuple[int, int, int]],
    text: PILTextRenderer,
) -> None:
    px = int(pacman_pos[0] * tile_size + tile_size // 2)
    py = int((pacman_pos[1] + 1) * tile_size) + 8
    msg = f"{pacman_reward:.1f}"
    surf = text.render(msg, colors.get("pacman", (255, 255, 0)))
    if surf is not None:
        r = surf.get_rect(center=(px, py))
        screen.blit(surf, r.topleft)
    for key, pos in ghost_positions.items():
        gx = int(pos[0] * tile_size + tile_size // 2)
        gy = int((pos[1] + 1) * tile_size) + 8
        val = ghost_rewards.get(key, 0.0)
        msg = f"{val:.1f}"
        color = colors.get(f"ghost_{key}", (200, 200, 200))
        surf = text.render(msg, color)
        if surf is not None:
            r = surf.get_rect(center=(gx, gy))
            screen.blit(surf, r.topleft)


def draw_ui_strip(
    screen: pygame.Surface,
    game_width: int,
    game_height: int,
    button_rects: list[tuple[pygame.Rect, str, int | None]],
    active_speed_index: int,
    text: PILTextRenderer,
    show_popups: bool = True,
    show_debug_rewards: bool = False,
    max_steps_index: int = 2,
    level_index: int = 0,
) -> None:
    strip_rect = pygame.Rect(0, game_height, game_width, UI_STRIP_HEIGHT)
    pygame.draw.rect(screen, (40, 40, 40), strip_rect)
    pygame.draw.line(
        screen, (80, 80, 80), (0, game_height), (game_width, game_height), 2
    )

    for rect, label, value in button_rects:
        if value == POPUPS_BUTTON_VALUE:
            is_active = show_popups
        elif value == DEBUG_REWARDS_BUTTON_VALUE:
            is_active = show_debug_rewards
        elif value is not None and ROUNDS_BUTTON_BASE <= value < ROUNDS_BUTTON_BASE + 10:
            is_active = (value - ROUNDS_BUTTON_BASE) == max_steps_index
        elif value is not None and value in (MAP_BUTTON_BASE + 1, MAP_BUTTON_BASE + 2):
            is_active = (value - MAP_BUTTON_BASE) == level_index + 1
        else:
            is_active = value is not None and value == active_speed_index
        color = (70, 120, 70) if is_active else (60, 60, 60)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (120, 120, 120), rect, 1)

        cached = text.get_cached_label(label)
        if cached is None:
            text.set_cached_label(label, text.render(label, (220, 220, 220)))
            cached = text.get_cached_label(label)
        if cached is not None:
            tr = cached.get_rect(center=rect.center)
            screen.blit(cached, tr)

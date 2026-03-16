from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pygame


LOG_DIR = Path("logs")
EVAL_LOG_FILE = LOG_DIR / "eval_runs.jsonl"


def _load_eval_data() -> Tuple[List[int], List[float], Dict[str, List[float]]]:
    """Читает весь eval_runs.jsonl и возвращает:

    - episodes: список номеров эпизодов
    - pacman_rewards: reward Pacman по eval-забегам
    - ghost_rewards_series: dict[ghost_key] -> список reward по тем же эпизодам
    """
    if not EVAL_LOG_FILE.exists():
        return [], [], {}

    episodes: List[int] = []
    pacman_rewards: List[float] = []
    ghost_rewards_series: Dict[str, List[float]] = {}

    try:
        with EVAL_LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("run_type") != "eval":
                    continue
                ep = obj.get("episode_num")
                pac = obj.get("pacman", {}).get("total_reward")
                ghosts = obj.get("ghosts", {})
                if not isinstance(ep, int) or not isinstance(pac, (int, float)):
                    continue
                episodes.append(ep)
                pacman_rewards.append(float(pac))
                for key, vals in ghosts.items():
                    total_r = vals.get("total_reward")
                    if not isinstance(total_r, (int, float)):
                        continue
                    series = ghost_rewards_series.setdefault(key, [])
                    # Чтобы длины совпадали с episodes, заполняем пропуски нулями
                    while len(series) < len(episodes) - 1:
                        series.append(0.0)
                    series.append(float(total_r))
        # Выравниваем все серии призраков по длине эпизодов
        for key, series in ghost_rewards_series.items():
            while len(series) < len(episodes):
                series.append(0.0)
    except OSError:
        return [], [], {}

    return episodes, pacman_rewards, ghost_rewards_series


def _draw_axes(
    screen: pygame.Surface,
    rect: pygame.Rect,
    episodes: List[int],
    values: List[float],
    color: Tuple[int, int, int],
    label: str,
    font: pygame.font.Font | None,
) -> None:
    """Рисует один линейный график в прямоугольнике rect."""
    pygame.draw.rect(screen, (20, 20, 30), rect)
    pygame.draw.rect(screen, (70, 70, 100), rect, 1)

    if not episodes or not values:
        if font is not None:
            text = font.render("Waiting for eval data...", True, (200, 200, 220))
            screen.blit(text, (rect.x + 8, rect.y + 8))
        return

    xs = episodes
    ys = values
    min_y = min(ys)
    max_y = max(ys)
    if max_y == min_y:
        max_y += 1.0

    pad_x = 8
    pad_y = 8
    w = rect.width - 2 * pad_x
    h = rect.height - 2 * pad_y
    if w <= 0 or h <= 0:
        return

    points: List[Tuple[int, int]] = []
    for i, y in enumerate(ys):
        t = i / max(1, len(ys) - 1)
        px = rect.x + pad_x + int(t * w)
        norm = (y - min_y) / (max_y - min_y)
        py = rect.y + pad_y + h - int(norm * h)
        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(screen, color, False, points, 2)
    else:
        pygame.draw.circle(screen, color, points[0], 2)

    if font is not None:
        label_text = f"{label}  min={min_y:.1f}  max={max_y:.1f}"
        text = font.render(label_text, True, (220, 220, 240))
        screen.blit(text, (rect.x + 8, rect.y + 4))


def live_plot(refresh_seconds: float = 1.0) -> None:
    """Простой live-плот на pygame: каждые N секунд перечитываем лог и обновляем графики."""
    pygame.init()
    width, height = 900, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pacman RL eval plots")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.SysFont("consolas", 16)
    except Exception:
        font = None

    last_reload_time = 0.0
    episodes: List[int] = []
    pacman_rewards: List[float] = []
    ghost_rewards_series: Dict[str, List[float]] = {}

    colors = {
        "background": (10, 10, 16),
        "pacman": (255, 215, 0),
        "ghost_blinky": (255, 80, 80),
        "ghost_pinky": (255, 105, 180),
        "ghost_inky": (80, 200, 255),
        "ghost_clyde": (255, 165, 0),
        "ghost_default": (200, 200, 200),
    }

    running = True
    while running:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if now - last_reload_time >= refresh_seconds:
            last_reload_time = now
            episodes, pacman_rewards, ghost_rewards_series = _load_eval_data()

        screen.fill(colors["background"])

        # Верхняя половина — Pacman
        top_rect = pygame.Rect(20, 20, width - 40, (height - 60) // 2)
        _draw_axes(
            screen,
            top_rect,
            episodes,
            pacman_rewards,
            colors["pacman"],
            "Pacman reward",
            font,
        )

        # Нижняя половина — суммарный график призраков (по сумме их reward)
        bottom_rect = pygame.Rect(
            20, top_rect.bottom + 20, width - 40, (height - 60) // 2
        )
        if episodes and ghost_rewards_series:
            # Для простоты — суммарный reward всех призраков на графике
            ghost_sum = [
                sum(series[i] for series in ghost_rewards_series.values())
                for i in range(len(episodes))
            ]
            _draw_axes(
                screen,
                bottom_rect,
                episodes,
                ghost_sum,
                colors["ghost_default"],
                "Ghosts total reward (sum)",
                font,
            )
        else:
            _draw_axes(
                screen,
                bottom_rect,
                [],
                [],
                colors["ghost_default"],
                "Ghosts total reward (sum)",
                font,
            )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    live_plot()


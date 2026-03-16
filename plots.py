from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pygame


LOG_DIR = Path("logs")
EVAL_LOG_FILE = LOG_DIR / "eval_runs.jsonl"


def _load_eval_data() -> Tuple[
    List[int],
    List[float],
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
    Dict[str, List[float]],
]:
    """Читает весь eval_runs.jsonl и возвращает:

    - episodes: список номеров эпизодов
    - pacman_rewards: reward Pacman по eval-забегам
    - ghost_rewards_series: dict[ghost_key] -> список reward по тем же эпизодам
    - pills_eaten: env["pills_eaten"] по эпизодам
    - power_pills_eaten: env["power_pills_eaten"] по эпизодам
    - ghost_hits: env["ghost_hits"] по эпизодам
    """
    if not EVAL_LOG_FILE.exists():
        return [], [], {}, {}, {}, {}

    episodes: List[int] = []
    pacman_rewards: List[float] = []
    ghost_rewards_series: Dict[str, List[float]] = {}
    pills_eaten: Dict[str, List[float]] = {"pills": []}
    power_pills_eaten: Dict[str, List[float]] = {"power_pills": []}
    ghost_hits: Dict[str, List[float]] = {"ghost_hits": []}

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
                env = obj.get("env", {})
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
                # Env-счётчики по эпизодам
                pills_val = env.get("pills_eaten")
                power_pills_val = env.get("power_pills_eaten")
                ghost_hits_val = env.get("ghost_hits")
                pills_eaten["pills"].append(float(pills_val) if isinstance(pills_val, (int, float)) else 0.0)
                power_pills_eaten["power_pills"].append(
                    float(power_pills_val) if isinstance(power_pills_val, (int, float)) else 0.0
                )
                ghost_hits["ghost_hits"].append(
                    float(ghost_hits_val) if isinstance(ghost_hits_val, (int, float)) else 0.0
                )
        # Выравниваем все серии призраков по длине эпизодов
        for key, series in ghost_rewards_series.items():
            while len(series) < len(episodes):
                series.append(0.0)
    except OSError:
        return [], [], {}, {}, {}, {}

    return episodes, pacman_rewards, ghost_rewards_series, pills_eaten, power_pills_eaten, ghost_hits


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
    width, height = 1100, 700
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
    pills_eaten: Dict[str, List[float]] = {"pills": []}
    power_pills_eaten: Dict[str, List[float]] = {"power_pills": []}
    ghost_hits: Dict[str, List[float]] = {"ghost_hits": []}

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
            (
                episodes,
                pacman_rewards,
                ghost_rewards_series,
                pills_eaten,
                power_pills_eaten,
                ghost_hits,
            ) = _load_eval_data()

        screen.fill(colors["background"])

        # Разбиваем окно на 2x2 панели
        col_w = (width - 60) // 2
        row_h = (height - 80) // 2

        # Лево-верх: Pacman reward
        top_left = pygame.Rect(20, 20, col_w, row_h)
        _draw_axes(
            screen,
            top_left,
            episodes,
            pacman_rewards,
            colors["pacman"],
            "Pacman reward",
            font,
        )

        # Право-верх: env.pills_eaten
        top_right = pygame.Rect(40 + col_w, 20, col_w, row_h)
        _draw_axes(
            screen,
            top_right,
            episodes,
            pills_eaten.get("pills", []),
            (120, 220, 120),
            "Env: pills_eaten",
            font,
        )

        # Лево-низ: отдельные призраки по reward (если есть)
        bottom_left = pygame.Rect(20, 40 + row_h, col_w, row_h)
        pygame.draw.rect(screen, (20, 20, 30), bottom_left)
        pygame.draw.rect(screen, (70, 70, 100), bottom_left, 1)
        if episodes and ghost_rewards_series:
            pad_x = 8
            pad_y = 8
            w = bottom_left.width - 2 * pad_x
            h = bottom_left.height - 2 * pad_y
            if w > 0 and h > 0:
                # Готовим общий min/max по всем сериям
                all_vals: List[float] = []
                for series in ghost_rewards_series.values():
                    all_vals.extend(series)
                if all_vals:
                    min_y = min(all_vals)
                    max_y = max(all_vals)
                    if max_y == min_y:
                        max_y += 1.0
                    # каждая серия своим цветом
                    for key, series in sorted(ghost_rewards_series.items()):
                        color = colors.get(f"ghost_{key}", colors["ghost_default"])
                        points: List[Tuple[int, int]] = []
                        for i, y in enumerate(series):
                            t = i / max(1, len(series) - 1)
                            px = bottom_left.x + pad_x + int(t * w)
                            norm = (y - min_y) / (max_y - min_y)
                            py = bottom_left.y + pad_y + h - int(norm * h)
                            points.append((px, py))
                        if len(points) >= 2:
                            pygame.draw.lines(screen, color, False, points, 2)
                        elif points:
                            pygame.draw.circle(screen, color, points[0], 2)
                    if font is not None:
                        label_text = f"Ghost rewards per agent  min={min_y:.1f} max={max_y:.1f}"
                        text = font.render(label_text, True, (220, 220, 240))
                        screen.blit(text, (bottom_left.x + 8, bottom_left.y + 4))
        else:
            if font is not None:
                text = font.render("Waiting for eval data...", True, (200, 200, 220))
                screen.blit(text, (bottom_left.x + 8, bottom_left.y + 8))

        # Право-низ: env.ghost_hits (сколько раз Pacman умер)
        bottom_right = pygame.Rect(40 + col_w, 40 + row_h, col_w, row_h)
        _draw_axes(
            screen,
            bottom_right,
            episodes,
            ghost_hits.get("ghost_hits", []),
            (220, 120, 120),
            "Env: ghost_hits (Pacman deaths)",
            font,
        )

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    live_plot()


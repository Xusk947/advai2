from __future__ import annotations

import pygame

from env.pacman_env import PacmanEnv

from .assets import load_ghost_sprites, load_pacman_sprites
from .constants import (
    DEBUG_REWARDS_BUTTON_VALUE,
    MAP_BUTTON_BASE,
    MAX_STEPS_PRESETS,
    POPUPS_BUTTON_VALUE,
    ROUNDS_BUTTON_BASE,
    SPEED_PRESETS,
    UI_STRIP_HEIGHT,
)
from .draw import (
    build_caption,
    draw_entity_rewards,
    draw_event_popups,
    draw_ghosts,
    draw_level,
    draw_pacman,
    draw_reward,
    draw_ui_strip,
)
from .text import PILTextRenderer
from .ui import build_ui_buttons


class PygameRenderer:
    def __init__(self, env: PacmanEnv, tile_size: int = 32) -> None:
        self.env = env
        self.tile_size = tile_size

        assert env.level is not None, "Call env.reset() before creating renderer."
        w, h = env.level.width, env.level.height
        self._game_height = h * tile_size
        self._game_width = w * tile_size

        pygame.init()
        self.screen = pygame.display.set_mode(
            (self._game_width, self._game_height + UI_STRIP_HEIGHT)
        )
        pygame.display.set_caption("Pacman RL")
        self.clock = pygame.time.Clock()

        self._text = PILTextRenderer()
        self._ghost_body_frames, self._eye_platform, self._eye_pupil = (
            load_ghost_sprites(tile_size)
        )
        self._pacman_frames = load_pacman_sprites(tile_size)
        self._ghost_tinted_cache: dict[
            tuple[str, int], pygame.Surface
        ] = {}  # (key or "frightened", frame_idx)

        self.colors = {
            "background": (0, 0, 0),
            "wall": (20, 20, 20),
            "pill": (255, 255, 255),
            "power_pill": (0, 0, 255),
            "pacman": (255, 255, 0),
            "ghost_blinky": (255, 0, 0),
            "ghost_pinky": (255, 105, 180),
            "ghost_inky": (0, 255, 255),
            "ghost_clyde": (255, 165, 0),
        }

        self._button_rects: list[tuple[pygame.Rect, str, int | None]] = []

    def close(self) -> None:
        pygame.quit()

    def set_env(self, env: PacmanEnv) -> None:
        """Replace env (e.g. after switching level); resizes window."""
        self.env = env
        assert env.level is not None
        w, h = env.level.width, env.level.height
        self._game_width = w * self.tile_size
        self._game_height = h * self.tile_size
        self.screen = pygame.display.set_mode(
            (self._game_width, self._game_height + UI_STRIP_HEIGHT)
        )

    def render(
        self,
        fps: int = 30,
        episode_num: int | None = None,
        episode_ended: bool = False,
        pacman_display: tuple[float, float] | None = None,
        ghost_display: dict[str, tuple[float, float]] | None = None,
        event_popups: list[tuple[float, float, str, tuple[int, int, int]]]
        | None = None,
        show_reward_popups: bool = True,
        show_debug_rewards: bool = False,
        pacman_reward: float = 0.0,
        ghost_rewards: dict[str, float] | None = None,
        max_steps: int | None = None,
        level_num: int | None = None,
        eval_history: list[tuple[int, float]] | None = None,
    ) -> dict:
        assert self.env.level is not None
        assert self.env.pacman is not None

        level = self.env.level
        pacman = self.env.pacman
        ghost_rewards = ghost_rewards or {}

        restart = False
        new_speed_index: int | None = None
        toggle_popups = False
        toggle_debug_rewards = False
        max_steps_down = False
        max_steps_up = False
        switch_level: int | None = None
        new_max_steps_index: int | None = None

        try:
            si = next(i for i, (f, _) in enumerate(SPEED_PRESETS) if f == fps)
        except StopIteration:
            si = 2
        try:
            max_steps_idx = (
                MAX_STEPS_PRESETS.index(max_steps)
                if max_steps is not None
                else 2
            )
        except ValueError:
            max_steps_idx = 2
        level_idx = (level_num - 1) if level_num is not None and level_num >= 1 else 0

        self._button_rects = build_ui_buttons(
            self._game_width,
            self._game_height,
            si,
            show_popups=show_reward_popups,
            show_debug_rewards=show_debug_rewards,
            max_steps_index=max_steps_idx,
            level_index=level_idx,
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    restart = True
                elif event.key == pygame.K_1:
                    switch_level = 1
                elif event.key == pygame.K_2:
                    switch_level = 2
                elif event.key == pygame.K_LEFTBRACKET:
                    max_steps_down = True
                elif event.key == pygame.K_RIGHTBRACKET:
                    max_steps_up = True
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for rect, _label, value in self._button_rects:
                    if rect.collidepoint(event.pos):
                        if value is None:
                            restart = True
                        elif value == POPUPS_BUTTON_VALUE:
                            toggle_popups = True
                        elif value == DEBUG_REWARDS_BUTTON_VALUE:
                            toggle_debug_rewards = True
                        elif value is not None and ROUNDS_BUTTON_BASE <= value < ROUNDS_BUTTON_BASE + len(MAX_STEPS_PRESETS):
                            new_max_steps_index = value - ROUNDS_BUTTON_BASE
                        elif value in (MAP_BUTTON_BASE + 1, MAP_BUTTON_BASE + 2):
                            switch_level = value - MAP_BUTTON_BASE
                        else:
                            new_speed_index = value
                        break

        self.screen.fill(self.colors["background"])
        draw_level(self.screen, level, self.tile_size, self.colors)
        draw_pacman(
            self.screen,
            pacman,
            self.tile_size,
            self.colors,
            pacman_display,
            pacman_frames=self._pacman_frames,
            steps_done=self.env.steps_done,
        )
        draw_ghosts(
            self.screen,
            self.env.ghosts,
            self.tile_size,
            self.colors,
            self._ghost_body_frames,
            self._eye_platform,
            self._eye_pupil,
            self._ghost_tinted_cache,
            self.env.steps_done,
            ghost_display,
        )

        if show_reward_popups and event_popups:
            draw_event_popups(self.screen, self.tile_size, event_popups, self._text)

        if (
            show_debug_rewards
            and pacman_display is not None
            and ghost_display is not None
        ):
            draw_entity_rewards(
                self.screen,
                self.tile_size,
                pacman_display,
                pacman_reward,
                ghost_display,
                ghost_rewards,
                self.colors,
                self._text,
            )

        steps = self.env.steps_done
        reward = self.env.score
        pygame.display.set_caption(
            build_caption(
                steps, reward, episode_num, episode_ended,
                max_steps=max_steps, level_num=level_num,
            )
        )
        draw_reward(self.screen, self._text, reward)
        draw_ui_strip(
            self.screen,
            self._game_width,
            self._game_height,
            self._button_rects,
            si,
            self._text,
            show_popups=show_reward_popups,
            show_debug_rewards=show_debug_rewards,
            max_steps_index=max_steps_idx,
            level_index=level_idx,
        )

        if eval_history:
            self._draw_eval_history(eval_history)

        pygame.display.flip()
        self.clock.tick(fps)
        return {
            "restart": restart,
            "speed_index": new_speed_index,
            "toggle_popups": toggle_popups,
            "toggle_debug_rewards": toggle_debug_rewards,
            "max_steps_down": max_steps_down,
            "max_steps_up": max_steps_up,
            "new_max_steps_index": new_max_steps_index,
            "switch_level": switch_level,
        }

    def _draw_eval_history(self, eval_history: list[tuple[int, float]]) -> None:
        """Рисует маленький график reward по eval-эпизодам в правом верхнем углу."""
        if not eval_history:
            return

        max_points = 100
        data = eval_history[-max_points:]
        rewards = [r for _, r in data]
        if not rewards:
            return

        chart_width = 220
        chart_height = 80
        margin = 8

        x0 = self._game_width - chart_width - margin
        y0 = margin + 24  # чуть ниже текста Reward

        chart_rect = pygame.Rect(x0, y0, chart_width, chart_height)
        pygame.draw.rect(self.screen, (15, 15, 25), chart_rect)
        pygame.draw.rect(self.screen, (80, 80, 100), chart_rect, 1)

        min_r = min(rewards)
        max_r = max(rewards)
        if max_r == min_r:
            max_r += 1.0

        def to_screen(i: int, r: float) -> tuple[int, int]:
            t = i / max(1, len(data) - 1)
            x = x0 + int(t * (chart_width - 10)) + 5
            norm = (r - min_r) / (max_r - min_r)
            y = y0 + chart_height - 6 - int(norm * (chart_height - 12))
            return x, y

        points = [to_screen(i, r) for i, r in enumerate(rewards)]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, (80, 200, 255), False, points, 2)
        else:
            pygame.draw.circle(self.screen, (80, 200, 255), points[0], 2)

        label = f"eval r: {min_r:.0f}..{max_r:.0f}"
        surf = self._text.render(label, (200, 220, 255))
        if surf is not None:
            self.screen.blit(surf, (x0 + 6, y0 + 4))

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .ghost_reward_shaping import get_all_ghost_shaping
from .pacman_reward_shaping import get_pacman_pill_shaping
from .ghosts.base import (
    ACTION_DELTAS,
    Ghost,
    GhostMode,
    _cell_walkable_for_ghost,
    _get_ghost_wrap_position,
)
from .level_loader import GridPos, LevelDefinition, LevelLoader
from .pacman import Direction, Pacman
from .vec2 import Vec2Int
from .reward_config import (
    ALL_PILLS,
    DEATH,
    GHOST_AT_HOME_OR_BARRIER,
    GHOST_CATCH,
    GHOST_EAT,
    GHOST_EATEN,
    GHOST_PILL,
    GHOST_POWER,
    GHOST_RESURRECTED,
    GHOST_TIME,
    POWER_PILL,
    POWER_UP_STEPS,
    PILL,
    STEP,
    STAY,
)

GHOST_KEYS_ORDER = ("blinky", "pinky", "inky", "clyde")


@dataclass
class StepResult:
    next_state: np.ndarray
    reward: float
    done: bool
    info: Dict


class PacmanEnv:
    def __init__(self, level_image_path: str, max_steps: int | None = 512) -> None:
        self.level_image_path = level_image_path
        self.max_steps = max_steps  # None = без лимита шагов

        self.level: LevelDefinition | None = None
        self.pacman: Pacman | None = None
        self.ghosts: Dict[str, Ghost] = {}
        self.steps_done: int = 0
        self.score: float = 0.0

        self.pills_eaten: int = 0
        self.power_pills_eaten: int = 0
        self.ghost_hits: int = 0
        self.ghosts_eaten: int = 0
        # Комбо множитель за поедание призраков под действием power pill
        self._ghost_combo_mult: float = 1.0

        # Кэш уровня: не перезагружаем картинку при каждом reset()
        self._level_template: LevelDefinition | None = None
        self._total_pills: int = 0  # для быстрой проверки _all_pills_eaten
        # Счётчик шагов призрака в домике/на барьере для накапливающегося штрафа
        self._ghost_home_steps: Dict[str, int] = {}

    @property
    def shape(self) -> Tuple[int, int, int]:
        assert self.level is not None
        return (5, self.level.height, self.level.width)

    def reset(self) -> np.ndarray:
        if self._level_template is None:
            loader = LevelLoader(self.level_image_path)
            self._level_template = loader.load()
        # Копируем только изменяемые поля (pills, power_pills), остальное по ссылке
        t = self._level_template
        self.level = LevelDefinition(
            width=t.width,
            height=t.height,
            walls=t.walls,
            pacman_barrier=t.pacman_barrier,
            pills=copy.deepcopy(t.pills),
            power_pills=copy.deepcopy(t.power_pills),
            no_pill_zone=t.no_pill_zone,
            pacman_start=t.pacman_start,
            ghost_starts=t.ghost_starts,
        )

        self.pacman = Pacman(pos=Vec2Int.from_tuple(self.level.pacman_start))
        self.ghosts = {
            key: Ghost(key=key, pos=Vec2Int.from_tuple(pos), home_pos=Vec2Int.from_tuple(pos))
            for key, pos in self.level.ghost_starts.items()
        }
        # При каждом reset() обнуляем счётчики «зависания» в домике
        self._ghost_home_steps = {key: 0 for key in self.ghosts}

        self._total_pills = sum(sum(1 for c in row if c) for row in self.level.pills) + sum(
            sum(1 for c in row if c) for row in self.level.power_pills
        )
        self.steps_done = 0
        self.score = 0.0

        self.pills_eaten = 0
        self.power_pills_eaten = 0
        self.ghost_hits = 0
        self.ghosts_eaten = 0
        self._ghost_combo_mult = 1.0

        return self._get_pacman_state_vector()

    def step(
        self,
        action: int,
        ghost_actions: Dict[str, int],
    ) -> StepResult:
        assert self.level is not None
        assert self.pacman is not None

        level = self.level
        pacman = self.pacman

        self.steps_done += 1
        reward = STEP

        old_pos = pacman.pos
        new_pos = pacman.move(action, level)
        if new_pos == old_pos:
            reward += STAY

        x, y = new_pos.x, new_pos.y
        pill_eaten_this_step = False
        power_eaten_this_step = False
        if level.pills[y][x]:
            level.pills[y][x] = False
            self.pills_eaten += 1
            pill_eaten_this_step = True
            reward += PILL
        if level.power_pills[y][x]:
            level.power_pills[y][x] = False
            self.power_pills_eaten += 1
            power_eaten_this_step = True
            pacman.powered_up_steps = POWER_UP_STEPS
            # Новая power pill — начинаем комбо заново
            self._ghost_combo_mult = 1.0
            for g in self.ghosts.values():
                if g.mode is not GhostMode.EATEN:
                    g.mode = GhostMode.FRIGHTENED
            reward += POWER_PILL

        ghost_old_positions = {key: ghost.pos for key, ghost in self.ghosts.items()}
        ghosts_eaten_before = {key for key, g in self.ghosts.items() if g.mode is GhostMode.EATEN}
        for key, ghost in self.ghosts.items():
            a = ghost_actions.get(key, 0)
            ghost.step_with_action(level, a)

        done = False
        pacman_died_this_step = False
        ghosts_eaten_this_step: list[str] = []  # кому дать штраф GHOST_EATEN — мотивация вернуться на базу
        ghosts_hit_pacman_this_step: list[str] = []
        ghost_combo_used: Dict[str, float] = {}

        for key, ghost in self.ghosts.items():
            # Одна клетка после хода — столкновение
            same_cell = ghost.pos == pacman.pos
            # Перекрёст: двигались навстречу и поменялись клетками (призрак старый = пакман новый, пакман старый = призрак новый)
            crossed = ghost_old_positions[key] == pacman.pos and old_pos == ghost.pos
            if same_cell or crossed:
                if pacman.powered_up_steps > 0 and ghost.mode is not GhostMode.EATEN:
                    ghost.mode = GhostMode.EATEN
                    self.ghosts_eaten += 1
                    ghosts_eaten_this_step.append(key)
                    # Комбо поедания призраков: каждый следующий под действием текущей power pill даёт в 2 раза больше
                    reward += GHOST_EAT * self._ghost_combo_mult
                    ghost_combo_used[key] = self._ghost_combo_mult
                    self._ghost_combo_mult *= 2.0
                elif ghost.mode is not GhostMode.EATEN:
                    self.ghost_hits += 1
                    ghosts_hit_pacman_this_step.append(key)
                    if not done:
                        reward += DEATH  # только один раз, даже если несколько призраков на клетке
                    done = True
                    pacman_died_this_step = True

        if pacman.powered_up_steps == 0:
            for ghost in self.ghosts.values():
                if ghost.mode is GhostMode.FRIGHTENED:
                    ghost.mode = GhostMode.NORMAL
        # Уменьшаем таймер силы после проверки столкновений, чтобы последний тик ещё давал поедание призрака
        pacman.powered_up_steps = max(0, pacman.powered_up_steps - 1)

        if self.pills_eaten + self.power_pills_eaten >= self._total_pills:
            reward += ALL_PILLS
            done = True

        if self.max_steps is not None and self.steps_done >= self.max_steps:
            done = True

        reward += get_pacman_pill_shaping(self)
        self.score += reward

        ghost_shared_reward = GHOST_TIME
        if pacman_died_this_step:
            ghost_shared_reward += GHOST_CATCH
        if pill_eaten_this_step:
            ghost_shared_reward += GHOST_PILL
        if power_eaten_this_step:
            ghost_shared_reward += GHOST_POWER

        # Награда за возврат на базу (был EATEN, после step стал NORMAL на home_pos)
        resurrected_this_step = [
            key
            for key, g in self.ghosts.items()
            if key in ghosts_eaten_before and g.mode is GhostMode.NORMAL and g.pos == g.home_pos
        ]
        shaping = get_all_ghost_shaping(self)
        # Штраф за сидение в домике/на барьере — только в NORMAL, чтобы выходили наружу
        at_home_or_barrier = {}
        if self.level is not None:
            h, w = self.level.height, self.level.width
            for key, g in self.ghosts.items():
                if g.mode is not GhostMode.NORMAL:
                    at_home_or_barrier[key] = 0.0
                    # вне NORMAL считаем, что «зависания» в домике нет
                    self._ghost_home_steps[key] = 0
                    continue

                is_home = g.pos == g.home_pos
                is_barrier = (
                    0 <= g.pos.y < h
                    and 0 <= g.pos.x < w
                    and self.level.pacman_barrier[g.pos.y][g.pos.x]
                )

                if is_home or is_barrier:
                    # Накапливаем шаги в домике/на барьере
                    prev = self._ghost_home_steps.get(key, 0)
                    cur = prev + 1
                    self._ghost_home_steps[key] = cur
                    # Каждые 2 шага усиливаем штраф в 1.2 раза
                    factor = 1.2 ** (cur // 2)
                    at_home_or_barrier[key] = GHOST_AT_HOME_OR_BARRIER * factor
                else:
                    # Вышел наружу — сбрасываем накопление
                    self._ghost_home_steps[key] = 0
                    at_home_or_barrier[key] = 0.0
        # Штраф GHOST_EATEN только съеденному призраку — мотивация вернуться на базу и «восстановиться».
        # При этом усиливаем штраф в зависимости от комбо Pacman (ghost_combo_used).
        ghost_rewards = {
            key: ghost_shared_reward
            + (
                GHOST_EATEN * ghost_combo_used.get(key, 1.0)
                if key in ghosts_eaten_this_step
                else 0.0
            )
            + (GHOST_RESURRECTED if key in resurrected_this_step else 0.0)
            + at_home_or_barrier.get(key, 0.0)
            + shaping.get(key, 0.0)
            for key in self.ghosts
        }

        next_state = self._get_pacman_state_vector()
        info = {
            "score": self.score,
            "steps": self.steps_done,
            "pills_eaten": self.pills_eaten,
            "power_pills_eaten": self.power_pills_eaten,
            "ghost_hits": self.ghost_hits,
            "ghosts_eaten": self.ghosts_eaten,
            "ghost_rewards": ghost_rewards,
            # Пер-степовые детали для логирования / анализа
            "ghosts_eaten_this_step": ghosts_eaten_this_step,
            "ghosts_hit_pacman_this_step": ghosts_hit_pacman_this_step,
        }
        return StepResult(next_state=next_state, reward=reward, done=done, info=info)

    def _ghost_scalars(self, ghost_key: str) -> np.ndarray:
        """Extra scalars for a ghost: Pacman power left, own direction one-hot, own mode, steps left, pills left."""
        assert self.pacman is not None
        ghost = self.ghosts[ghost_key]
        scalars: list[float] = []

        power_norm = self.pacman.powered_up_steps / POWER_UP_STEPS if POWER_UP_STEPS else 0.0
        scalars.append(min(1.0, max(0.0, power_norm)))

        direction_onehot = [
            1.0 if ghost.direction == 0 else 0.0,
            1.0 if ghost.direction == 1 else 0.0,
            1.0 if ghost.direction == 2 else 0.0,
            1.0 if ghost.direction == 3 else 0.0,
        ]
        scalars.extend(direction_onehot)

        if ghost.mode is GhostMode.NORMAL:
            scalars.append(1.0)
        elif ghost.mode is GhostMode.FRIGHTENED:
            scalars.append(0.7)
        else:
            scalars.append(0.3)

        if self.max_steps is not None and self.max_steps > 0:
            steps_left = max(0, self.max_steps - self.steps_done)
            scalars.append(steps_left / self.max_steps)
        else:
            scalars.append(1.0)

        pills_remaining = self._total_pills - (self.pills_eaten + self.power_pills_eaten)
        pills_norm = pills_remaining / self._total_pills if self._total_pills else 0.0
        scalars.append(min(1.0, max(0.0, pills_norm)))
        scalars.extend(self._ghost_valid_moves(ghost_key).tolist())
        return np.array(scalars, dtype=np.float32)

    def get_ghost_state(self, ghost_key: str) -> np.ndarray:
        """State for a ghost agent: (5,H,W) flattened + one-hot ghost id (4) + scalars (12), shape (5*H*W + 16,).
        C0 = только стены (без pacman_barrier), чтобы призрак «видел» выход из домика как проходимый."""
        state = self._build_state(include_barrier_as_wall=False)
        flat = state.flatten().astype(np.float32)
        one_hot = np.zeros(4, dtype=np.float32)
        if ghost_key in GHOST_KEYS_ORDER:
            idx = GHOST_KEYS_ORDER.index(ghost_key)
            one_hot[idx] = 1.0
        scalars = self._ghost_scalars(ghost_key)
        return np.concatenate([flat, one_hot, scalars])

    def get_all_ghost_states(self) -> Dict[str, np.ndarray]:
        """Строит состояние один раз и возвращает вектор для каждого призрака (оптимизация вместо 4x get_ghost_state)."""
        state = self._build_state(include_barrier_as_wall=False)
        flat = state.flatten().astype(np.float32)
        out = {}
        for key in self.ghosts:
            one_hot = np.zeros(4, dtype=np.float32)
            if key in GHOST_KEYS_ORDER:
                idx = GHOST_KEYS_ORDER.index(key)
                one_hot[idx] = 1.0
            scalars = self._ghost_scalars(key)
            out[key] = np.concatenate([flat, one_hot, scalars])
        return out

    def _build_state(self, include_barrier_as_wall: bool = True) -> np.ndarray:
        assert self.level is not None
        assert self.pacman is not None

        level = self.level
        h, w = level.height, level.width
        state = np.zeros((5, h, w), dtype=np.float32)

        # C0: для Pacman — стены + барьер (барьер непроходим); для призраков — только стены (барьер проходим, выход из домика виден)
        if include_barrier_as_wall:
            c0 = np.logical_or(
                np.array(level.walls, dtype=bool),
                np.array(level.pacman_barrier, dtype=bool),
            )
        else:
            c0 = np.array(level.walls, dtype=bool)
        state[0] = c0.astype(np.float32)

        # C1, regular pills
        state[1] = np.array(level.pills, dtype=np.float32)

        # C2, power pills
        state[2] = np.array(level.power_pills, dtype=np.float32)

        # C3, Pacman
        state[3, self.pacman.pos.y, self.pacman.pos.x] = 1.0

        # C4, ghosts: 1.0 = NORMAL (dangerous), 0.7 = FRIGHTENED (edible), 0.3 = EATEN (safe, eyes only)
        for ghost in self.ghosts.values():
            if 0 <= ghost.pos.x < w and 0 <= ghost.pos.y < h:
                if ghost.mode is GhostMode.NORMAL:
                    state[4, ghost.pos.y, ghost.pos.x] = 1.0
                elif ghost.mode is GhostMode.FRIGHTENED:
                    state[4, ghost.pos.y, ghost.pos.x] = 0.7
                else:
                    state[4, ghost.pos.y, ghost.pos.x] = 0.3

        return state

    def _pacman_valid_moves(self) -> np.ndarray:
        """Маска ходов Pacman: [can_up, can_down, can_left, can_right], 1.0 если можно, 0.0 если стена/барьер."""
        assert self.level is not None and self.pacman is not None
        level = self.level
        pos = self.pacman.pos
        w, h = level.width, level.height
        out = []
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            next_pos = Vec2Int(pos.x + dx, pos.y + dy)
            if 0 <= next_pos.x < w and 0 <= next_pos.y < h:
                can = Pacman._can_move_to(next_pos, level)
            else:
                can = Pacman._get_wrap_position(next_pos, level) is not None
            out.append(1.0 if can else 0.0)
        return np.array(out, dtype=np.float32)

    def _ghost_valid_moves(self, ghost_key: str) -> np.ndarray:
        """Маска ходов призрака: [can_up, can_down, can_left, can_right], 1.0 если можно, 0.0 если стена."""
        assert self.level is not None
        level = self.level
        ghost = self.ghosts[ghost_key]
        pos = ghost.pos
        w, h = level.width, level.height
        out = []
        for i in range(4):
            next_pos = pos + ACTION_DELTAS[i]
            if 0 <= next_pos.x < w and 0 <= next_pos.y < h:
                can = _cell_walkable_for_ghost(level, next_pos)
            else:
                can = _get_ghost_wrap_position(level, next_pos) is not None
            out.append(1.0 if can else 0.0)
        return np.array(out, dtype=np.float32)

    def _pacman_scalars(self) -> np.ndarray:
        """Extra scalars for Pacman: power, direction one-hot, steps left, pills left, valid_moves (4)."""
        assert self.pacman is not None
        pacman = self.pacman
        scalars: list[float] = []

        power_norm = pacman.powered_up_steps / POWER_UP_STEPS if POWER_UP_STEPS else 0.0
        scalars.append(min(1.0, max(0.0, power_norm)))

        direction_onehot = [
            1.0 if pacman.direction is Direction.UP else 0.0,
            1.0 if pacman.direction is Direction.DOWN else 0.0,
            1.0 if pacman.direction is Direction.LEFT else 0.0,
            1.0 if pacman.direction is Direction.RIGHT else 0.0,
        ]
        scalars.extend(direction_onehot)

        if self.max_steps is not None and self.max_steps > 0:
            steps_left = max(0, self.max_steps - self.steps_done)
            scalars.append(steps_left / self.max_steps)
        else:
            scalars.append(1.0)

        pills_remaining = self._total_pills - (self.pills_eaten + self.power_pills_eaten)
        pills_norm = pills_remaining / self._total_pills if self._total_pills else 0.0
        scalars.append(min(1.0, max(0.0, pills_norm)))
        scalars.extend(self._pacman_valid_moves().tolist())
        return np.array(scalars, dtype=np.float32)

    def _get_pacman_state_vector(self) -> np.ndarray:
        """Full Pacman state: (5,H,W) flattened + scalars (power, direction, steps left, pills left, valid_moves 4)."""
        spatial = self._build_state().flatten().astype(np.float32)
        scalars = self._pacman_scalars()
        return np.concatenate([spatial, scalars])


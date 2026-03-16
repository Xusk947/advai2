from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


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


def live_plot(refresh_seconds: float = 5.0) -> None:
    """Простой live-плот: каждые N секунд перечитываем лог и обновляем графики."""
    plt.ion()
    fig, (ax_pacman, ax_ghosts) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    fig.suptitle("Pacman RL eval metrics (from logs/eval_runs.jsonl)")

    last_len = -1

    try:
        while True:
            episodes, pacman_rewards, ghost_rewards_series = _load_eval_data()
            if len(episodes) == 0:
                ax_pacman.clear()
                ax_ghosts.clear()
                ax_pacman.set_title("Waiting for eval data...")
                ax_pacman.set_ylabel("Pacman reward")
                ax_ghosts.set_ylabel("Ghost reward")
                ax_ghosts.set_xlabel("Episode")
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(refresh_seconds)
                continue

            # Обновляем графики только если появились новые точки
            if len(episodes) != last_len:
                last_len = len(episodes)
                ax_pacman.clear()
                ax_ghosts.clear()

                ax_pacman.plot(episodes, pacman_rewards, label="Pacman reward", color="gold")
                ax_pacman.set_ylabel("Pacman reward")
                ax_pacman.legend(loc="upper left")

                for key, series in sorted(ghost_rewards_series.items()):
                    ax_ghosts.plot(episodes, series, label=f"{key} reward")
                ax_ghosts.set_ylabel("Ghost reward")
                ax_ghosts.set_xlabel("Episode")
                ax_ghosts.legend(loc="upper left")

                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    live_plot()


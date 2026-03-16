from __future__ import annotations

"""Все основные коэффициенты наград/штрафов для среды Pacman."""

# Pacman: знак + награда, − штраф
STEP: float = -0.1
STAY: float = -0.5  # остался на той же клетке
PILL: float = 1.5  # чуть выше, чтобы охотнее шёл за пилюлями
POWER_PILL: float = 5.0
POWER_UP_STEPS: int = 40
GHOST_EAT: float = 10.0
DEATH: float = -25.0
ALL_PILLS: float = 50.0

# Призраки: то же (+, −)
GHOST_TIME: float = -0.1
GHOST_CATCH: float = 40.0
GHOST_PILL: float = -0.5
GHOST_POWER: float = -2.0
GHOST_EATEN: float = -7.0
GHOST_RESURRECTED: float = 2.0  # небольшая награда за возврат на базу — подкрепляет мотивацию вернуться
# Штраф за то, что призрак в домике или на клетке, куда Pacman не может — чтобы не засиживались
GHOST_AT_HOME_OR_BARRIER: float = -0.05

# Shaping призраков
GHOST_CLUSTER_PENALTY: float = -0.3
GLOBAL_CHASE_COEFF: float = -0.005

# Shaping Pacman по расстоянию до ближайшей пилюли
PACMAN_PILL_DIST_COEFF: float = 0.015


from __future__ import annotations

"""
Backwards-compatible re-export.

Реальная реализация рендера вынесена в `game/renderer/`, чтобы этот файл оставался коротким.
"""

from game.renderer import PygameRenderer, SPEED_PRESETS

__all__ = ["PygameRenderer", "SPEED_PRESETS"]


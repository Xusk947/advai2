from __future__ import annotations

from pathlib import Path


def _get_output_root() -> Path:
    """
    Базовая директория для артефактов (чекпоинты, логи и т.п.).
    - В Kaggle: /kaggle/working/output
    - Локально: ./output
    """
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        base = kaggle_working / "output"
    else:
        base = Path("output")
    base.mkdir(parents=True, exist_ok=True)
    return base


OUTPUT_ROOT = _get_output_root()
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
LOG_DIR = OUTPUT_ROOT / "logs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


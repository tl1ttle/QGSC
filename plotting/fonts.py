from pathlib import Path
import os
from matplotlib import font_manager


def times_path():
    configured = os.environ.get("QGSC_TIMES_FONT")
    candidate = Path(configured) if configured else Path("C:/Windows/Fonts/times.ttf")
    if candidate.is_file():
        return str(candidate)
    return font_manager.findfont("Times New Roman", fallback_to_default=True)


def register_times():
    font_manager.fontManager.addfont(times_path())

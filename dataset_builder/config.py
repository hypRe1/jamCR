import os
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPOSITORY_ROOT / ".env"

load_dotenv(ENV_FILE)


def get_asset_dir(asset_dir: str | Path | None = None) -> Path:
    """Return the configured Clash Royale asset directory."""

    configured_dir = asset_dir or os.environ.get("CLASH_ASSETS_DIR")
    if not configured_dir:
        raise RuntimeError(
            "CLASH_ASSETS_DIR is not configured. Set it in "
            f"{ENV_FILE} or pass asset_dir explicitly."
        )

    path = Path(configured_dir).expanduser()
    if not path.is_absolute():
        path = (REPOSITORY_ROOT / path).resolve()
    else:
        path = path.resolve()

    if not path.is_dir():
        raise FileNotFoundError(
            f"Clash Royale asset directory does not exist: {path}\n"
            "Set CLASH_ASSETS_DIR to the directory containing "
            "troops.json, buildings.json, towers/, and arenas/."
        )

    return path

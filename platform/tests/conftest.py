import sys
from pathlib import Path

PLATFORM_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PLATFORM_DIR.parent
for p in (str(PLATFORM_DIR), str(REPO_ROOT / "ml" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

RAW = PLATFORM_DIR / "data" / "raw"

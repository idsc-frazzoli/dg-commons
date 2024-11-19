from pathlib import Path

REPO_DIR: Path = Path(__file__).parent.parent.parent
OUT_TESTS_DIR: str = str(REPO_DIR / "out/test-results")

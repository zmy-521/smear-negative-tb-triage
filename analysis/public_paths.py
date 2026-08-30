"""Portable path configuration for the public analysis scripts.

The original private analysis used local absolute paths. Those paths have been
removed. Set TB_WORK_ROOT to a private workspace containing the source data
files described in docs/PRIVATE_DATA.md. No patient-level data are included in
this repository.
"""
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("TB_WORK_ROOT", REPO_ROOT / "private_workspace")).resolve()
FONT_PATH = os.environ.get("TB_FONT_PATH", "DejaVuSans-Bold.ttf")

# Legacy analysis folders are preserved underneath the configurable work root so
# that the archived scripts retain their original dependency graph.
AIM2_OUT = WORK_ROOT / "结核-Aim2" / "outputs"
EXT_OUT = WORK_ROOT / "结核-外部验证" / "outputs"
AIM2_OUT.mkdir(parents=True, exist_ok=True)
EXT_OUT.mkdir(parents=True, exist_ok=True)

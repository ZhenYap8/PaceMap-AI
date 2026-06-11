"""
Central path configuration for the PaceMap AI application.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LIBRARY_DIR = DATA_DIR / "library"
GENERATED_ROUTES_DIR = DATA_DIR / "generated_routes"
MODELS_DIR = PROJECT_ROOT / "models"
PROFILE_PATH = DATA_DIR / "runner_profile.json"
RUNS_DATA_PATH = DATA_DIR / "runs_data.json"

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

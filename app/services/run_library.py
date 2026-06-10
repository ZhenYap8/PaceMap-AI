"""
Persistent GPX run library for route recommendation.
"""

import json
import shutil
from pathlib import Path
from typing import Any

LIBRARY_DIR_NAME = "library"
INDEX_FILENAME = "index.json"


def get_library_dir(project_root: Path) -> Path:
    library_dir = project_root / "data" / LIBRARY_DIR_NAME
    library_dir.mkdir(parents=True, exist_ok=True)
    return library_dir


def _index_path(library_dir: Path) -> Path:
    return library_dir / INDEX_FILENAME


def load_index(library_dir: Path) -> list[dict[str, Any]]:
    index_path = _index_path(library_dir)
    if not index_path.exists():
        return []
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def save_index(library_dir: Path, index: list[dict[str, Any]]) -> None:
    with open(_index_path(library_dir), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def add_runs(library_dir: Path, uploaded_files: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    """Save uploaded GPX files and update the library index."""
    index = load_index(library_dir)
    existing_names = {entry["filename"] for entry in index}
    added: list[dict[str, Any]] = []

    for original_name, temp_path in uploaded_files:
        safe_name = Path(original_name).name
        if safe_name in existing_names:
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            counter = 1
            while f"{stem}_{counter}{suffix}" in existing_names:
                counter += 1
            safe_name = f"{stem}_{counter}{suffix}"

        dest_path = library_dir / safe_name
        shutil.copy2(temp_path, dest_path)

        entry = {
            "run_id": Path(safe_name).stem,
            "filename": safe_name,
            "filepath": str(dest_path),
        }
        index.append(entry)
        existing_names.add(safe_name)
        added.append(entry)

    save_index(library_dir, index)
    return added


def list_runs(library_dir: Path) -> list[dict[str, Any]]:
    return load_index(library_dir)


def clear_library(library_dir: Path) -> None:
    for gpx_file in library_dir.glob("*.gpx"):
        gpx_file.unlink()
    save_index(library_dir, [])


def reset_all_data(project_root: Path) -> dict[str, int]:
    """Clear library, learned profile, ML model, and generated routes."""
    library_dir = get_library_dir(project_root)
    cleared_runs = len(list_runs(library_dir))
    clear_library(library_dir)

    data_dir = project_root / "data"
    for filename in ("runner_profile.json", "runs_data.json"):
        path = data_dir / filename
        if path.exists():
            path.unlink()

    models_dir = project_root / "models"
    model_path = models_dir / "runner_model.pkl"
    if model_path.exists():
        model_path.unlink()

    generated_dir = data_dir / "generated_routes"
    cleared_generated = 0
    if generated_dir.exists():
        for gpx in generated_dir.glob("*.gpx"):
            gpx.unlink()
            cleared_generated += 1

    return {
        "cleared_runs": cleared_runs,
        "cleared_generated_routes": cleared_generated,
    }

"""
PaceMap AI — FastAPI backend serving API + frontend on port 8000.
"""

import json
import shutil
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.services.analysis import analyze_run
from app.services.run_library import add_runs, get_library_dir, list_runs, reset_all_data
from app.services.run_profile import learn_from_runs, load_profile
from app.services.route_recommender import recommend_route

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
OUTPUT_DIR = PROJECT_ROOT / "output"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
LIBRARY_DIR = get_library_dir(PROJECT_ROOT)
MODELS_DIR = PROJECT_ROOT / "models"
PROFILE_PATH = PROJECT_ROOT / "data" / "runner_profile.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PaceMap AI",
    description="AI-powered running performance analysis and route recommendation",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class RecommendRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    distance_km: float = Field(..., gt=0, le=200)
    smoothing: int = Field(default=3, ge=1, le=10)


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    profile = load_profile(PROFILE_PATH)
    return {
        "status": "ok",
        "service": "PaceMap AI",
        "library_runs": len(list_runs(LIBRARY_DIR)),
        "profile_learned": profile is not None,
    }


@app.post("/api/analyze")
async def analyze_gpx(
    file: UploadFile = File(...),
    smoothing: int = 3,
):
    """Upload a GPX file and run single-run analysis."""
    if not file.filename or not file.filename.lower().endswith(".gpx"):
        raise HTTPException(status_code=400, detail="Please upload a .gpx file")

    run_id = uuid.uuid4().hex[:8]
    safe_name = Path(file.filename).stem
    upload_path = UPLOAD_DIR / f"{safe_name}_{run_id}.gpx"

    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_run(
            gpx_filepath=str(upload_path),
            output_dir=str(OUTPUT_DIR),
            smoothing_window=smoothing,
        )
        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="GPX file not found")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to analyze GPX: {exc}")
    finally:
        if upload_path.exists():
            upload_path.unlink()


@app.post("/api/runs/upload")
async def upload_runs(files: list[UploadFile] = File(...)):
    """Upload multiple GPX files to the run library."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploaded: list[tuple[str, Path]] = []
    temp_paths: list[Path] = []

    try:
        for file in files:
            if not file.filename or not file.filename.lower().endswith(".gpx"):
                continue

            temp_path = UPLOAD_DIR / f"lib_{uuid.uuid4().hex[:8]}.gpx"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded.append((file.filename, temp_path))
            temp_paths.append(temp_path)

        if not uploaded:
            raise HTTPException(status_code=400, detail="No valid .gpx files found")

        added = add_runs(LIBRARY_DIR, uploaded)
        return {
            "added": len(added),
            "total_runs": len(list_runs(LIBRARY_DIR)),
            "runs": added,
        }
    finally:
        for temp_path in temp_paths:
            if temp_path.exists():
                temp_path.unlink()


@app.post("/api/runs/learn")
async def learn_runs(smoothing: int = 3):
    """Learn patterns from all runs in the library and train ML model."""
    entries = list_runs(LIBRARY_DIR)
    if not entries:
        raise HTTPException(
            status_code=400,
            detail="No runs in library. Upload GPX files first.",
        )

    try:
        profile = learn_from_runs(
            run_entries=entries,
            models_dir=MODELS_DIR,
            profile_path=PROFILE_PATH,
            smoothing_window=smoothing,
        )
        return profile
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Learning failed: {exc}")


@app.post("/api/runs/reset")
def reset_runs():
    """Clear the run library, learned profile, and trained model."""
    result = reset_all_data(PROJECT_ROOT)
    return {
        "reset": True,
        "message": "Library and profile cleared.",
        **result,
        "library_runs": 0,
    }


@app.get("/api/runs/profile")
def get_profile():
    """Get the current runner profile (after learning)."""
    profile = load_profile(PROFILE_PATH)
    if not profile:
        return {
            "learned": False,
            "library_runs": len(list_runs(LIBRARY_DIR)),
            "message": "Upload GPX files and click 'Learn My Runs' to build your profile.",
        }
    return {"learned": True, **profile}


@app.get("/api/geocode")
def geocode_location(q: str):
    """Geocode a location name to lat/lon using OpenStreetMap Nominatim."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Location query is required")

    params = urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})
    url = f"https://nominatim.openstreetmap.org/search?{params}"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PaceMap-AI/2.0 (running route recommender)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            results = json.loads(response.read().decode())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {exc}")

    if not results:
        raise HTTPException(status_code=404, detail=f"Location not found: {q}")

    place = results[0]
    return {
        "name": place.get("display_name", q),
        "latitude": float(place["lat"]),
        "longitude": float(place["lon"]),
    }


@app.post("/api/recommend")
async def recommend(request: RecommendRequest):
    """Generate and recommend a new route near the given location."""
    profile = load_profile(PROFILE_PATH)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail="No runner profile found. Upload GPX files and learn your runs first.",
        )

    historical_runs = []
    runs_data_path = PROFILE_PATH.parent / "runs_data.json"
    if runs_data_path.exists():
        with open(runs_data_path, encoding="utf-8") as f:
            historical_runs = json.load(f)

    gpx_dir = PROJECT_ROOT / "data" / "generated_routes"

    try:
        return recommend_route(
            profile=profile,
            models_dir=MODELS_DIR,
            output_dir=OUTPUT_DIR,
            gpx_dir=gpx_dir,
            target_lat=request.latitude,
            target_lon=request.longitude,
            target_distance_km=request.distance_km,
            historical_runs=historical_runs,
            smoothing_window=request.smoothing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Recommendation failed: {exc}")


@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)

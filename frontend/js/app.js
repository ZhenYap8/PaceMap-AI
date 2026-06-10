// ─── Shared ───────────────────────────────────────────────────────────────────

/** Display pace as M:SS/km (backend stores seconds internally). */
function formatPaceLabel(paceStr) {
  if (!paceStr) return null;
  const minSec = paceStr.replace(/\s*\/km$/i, "").trim();
  return `${minSec}/km pace`;
}

function formatPaceFromSeconds(sec) {
  if (!sec || sec <= 0) return "--:--/km";
  const total = Math.floor(sec);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, "0")}/km`;
}

function formatDurationFromSeconds(sec) {
  const total = Math.max(0, Math.floor(sec));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Fix legacy reason strings that still contain raw seconds. */
function humanizeCleaningReason(reason, run) {
  let text = reason;
  text = text.replace(
    /\((\d+(?:\.\d+)?)s\/km\)/g,
    (_, sec) => `(${formatPaceFromSeconds(Number(sec))})`
  );
  text = text.replace(
    /Finish time too short \((\d+(?:\.\d+)?)s\)/g,
    (_, sec) => `Finish time too short (${formatDurationFromSeconds(Number(sec))})`
  );
  if (run.avg_pace) {
    const pace = `${run.avg_pace.replace(/\s*\/km$/i, "").trim()}/km`;
    text = text.replace(/^Avg pace \(z=/, `Avg pace ${pace} (z=`);
  }
  return text;
}

const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");

let geocodedLat = null;
let geocodedLon = null;
let libraryRunCount = 0;

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      const data = await res.json();
      statusBadge.classList.add("online");
      statusText.textContent = "Server online";
      libraryRunCount = data.library_runs || 0;
      updateLibraryStatus();
    } else {
      throw new Error("Unhealthy");
    }
  } catch {
    statusText.textContent = "Server offline";
  }
}

function updateLibraryStatus() {
  const el = document.getElementById("library-count");
  const learnBtn = document.getElementById("learn-btn");
  if (el) {
    el.textContent = `${libraryRunCount} run${libraryRunCount !== 1 ? "s" : ""} in library`;
  }
  if (learnBtn) {
    learnBtn.disabled = libraryRunCount === 0;
  }
}

// ─── Tab Navigation ───────────────────────────────────────────────────────────

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
  });
});

// ─── Tab 1: Analyze Single Run ────────────────────────────────────────────────

const uploadSection = document.getElementById("upload-section");
const loadingSection = document.getElementById("loading-section");
const errorSection = document.getElementById("error-section");
const resultsSection = document.getElementById("results-section");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const smoothingInput = document.getElementById("smoothing");
const smoothingValue = document.getElementById("smoothing-value");
const errorMessage = document.getElementById("error-message");
const retryBtn = document.getElementById("retry-btn");
const newRunBtn = document.getElementById("new-run-btn");
const runTitle = document.getElementById("run-title");
const statsGrid = document.getElementById("stats-grid");
const paceChart = document.getElementById("pace-chart");
const paceMap = document.getElementById("pace-map");

function showAnalyzeSection(section) {
  [uploadSection, loadingSection, errorSection, resultsSection].forEach((el) => {
    el.classList.add("hidden");
  });
  section.classList.remove("hidden");
}

function resetToUpload() {
  fileInput.value = "";
  showAnalyzeSection(uploadSection);
}

function renderStats(container, stats, extra = []) {
  const items = [
    { label: "Distance", value: stats.distance },
    { label: "Duration", value: stats.duration },
    { label: "Avg Pace", value: stats.avg_pace },
    { label: "Elevation Gain", value: `${stats.elevation_gain_m} m` },
    { label: "Fastest Pace", value: stats.fastest_pace || "—" },
    { label: "Slowest Pace", value: stats.slowest_pace || "—" },
    ...extra,
  ];

  container.innerHTML = items
    .map(
      (item) => `
      <div class="stat-card">
        <div class="stat-label">${item.label}</div>
        <div class="stat-value">${item.value}</div>
      </div>
    `
    )
    .join("");
}

function renderAnalyzeResults(data) {
  runTitle.textContent = data.run_name;
  renderStats(statsGrid, data.stats, [
    { label: "GPS Points", value: data.stats.gps_points },
    { label: "Segments", value: data.stats.segments },
  ]);
  paceChart.src = `${data.outputs.pace_chart}?t=${Date.now()}`;
  paceMap.src = "about:blank";
  paceMap.src = `${data.outputs.pace_map}?t=${Date.now()}`;
  showAnalyzeSection(resultsSection);
}

async function analyzeFile(file) {
  if (!file.name.toLowerCase().endsWith(".gpx")) {
    errorMessage.textContent = "Please upload a .gpx file.";
    showAnalyzeSection(errorSection);
    return;
  }

  showAnalyzeSection(loadingSection);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`/api/analyze?smoothing=${smoothingInput.value}`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed.");
    renderAnalyzeResults(data);
  } catch (err) {
    errorMessage.textContent = err.message || "Something went wrong.";
    showAnalyzeSection(errorSection);
  }
}

uploadZone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) analyzeFile(fileInput.files[0]); });
uploadZone.addEventListener("dragover", (e) => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files[0]) analyzeFile(e.dataTransfer.files[0]);
});
smoothingInput.addEventListener("input", () => { smoothingValue.textContent = smoothingInput.value; });
retryBtn.addEventListener("click", resetToUpload);
newRunBtn.addEventListener("click", resetToUpload);

// ─── Tab 2: Route Recommendation ──────────────────────────────────────────────

const recUploadZone = document.getElementById("rec-upload-zone");
const recFileInput = document.getElementById("rec-file-input");
const recBrowseBtn = document.getElementById("rec-browse-btn");
const learnBtn = document.getElementById("learn-btn");
const recLoadingSection = document.getElementById("rec-loading-section");
const recLoadingText = document.getElementById("rec-loading-text");
const profileSection = document.getElementById("profile-section");
const profileStats = document.getElementById("profile-stats");
const runsList = document.getElementById("runs-list");
const recommendForm = document.getElementById("recommend-form");
const recErrorSection = document.getElementById("rec-error-section");
const recErrorMessage = document.getElementById("rec-error-message");
const recommendResults = document.getElementById("recommend-results");
const locationInput = document.getElementById("location-input");
const geocodeBtn = document.getElementById("geocode-btn");
const geocodeResult = document.getElementById("geocode-result");
const distanceInput = document.getElementById("distance-input");
const recommendBtn = document.getElementById("recommend-btn");

function showRecError(msg) {
  recErrorMessage.textContent = msg;
  recErrorSection.classList.remove("hidden");
}

function hideRecError() {
  recErrorSection.classList.add("hidden");
}

async function uploadLibraryFiles(files) {
  const gpxFiles = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".gpx"));
  if (!gpxFiles.length) {
    showRecError("Please upload .gpx files.");
    return;
  }

  hideRecError();
  recLoadingSection.classList.remove("hidden");
  recLoadingText.textContent = `Uploading ${gpxFiles.length} file${gpxFiles.length > 1 ? "s" : ""}...`;

  const formData = new FormData();
  gpxFiles.forEach((f) => formData.append("files", f));

  try {
    const res = await fetch("/api/runs/upload", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    libraryRunCount = data.total_runs;
    updateLibraryStatus();
  } catch (err) {
    showRecError(err.message);
  } finally {
    recLoadingSection.classList.add("hidden");
  }
}

function renderProfile(profile) {
  const uploaded = profile.total_runs_uploaded ?? profile.total_runs;
  const used = profile.runs_used_for_learning ?? profile.total_runs;
  const cleaning = profile.outlier_cleaning;

  const items = [
    { label: "Runs Used", value: `${used} of ${uploaded}` },
    { label: "Avg Distance", value: `${profile.avg_distance_km} km` },
    { label: "Avg Pace", value: profile.avg_pace },
    { label: "Avg Elevation", value: `${profile.avg_elevation_gain_m} m` },
    { label: "Avg Finish Time", value: profile.avg_finish_time },
    { label: "Avg Heart Rate", value: `${profile.avg_heart_rate_bpm} bpm` },
  ];

  if (profile.avg_cadence_spm) {
    items.push({ label: "Avg Cadence", value: `${profile.avg_cadence_spm} spm` });
  }
  if (profile.model_trained) {
    items.push({ label: "ML Model R²", value: profile.model_metrics.r2 });
  }

  profileStats.innerHTML = items
    .map(
      (item) => `
      <div class="stat-card">
        <div class="stat-label">${item.label}</div>
        <div class="stat-value">${item.value}</div>
      </div>
    `
    )
    .join("");

  let cleaningHtml = "";
  if (cleaning && cleaning.removed_count > 0) {
    const stageBlocks = (cleaning.stages || [])
      .filter((s) => s.removed_count > 0)
      .map(
        (s) => `
        <div class="cleaning-stage">
          <h4>${s.description || s.stage}</h4>
          <ul>${(s.ignored_runs || []).map((o) => {
            const metaParts = [
              formatPaceLabel(o.avg_pace),
              o.finish_time ? `${o.finish_time} finish` : null,
            ].filter(Boolean);
            const meta = metaParts.length ? ` (${metaParts.join(" · ")})` : "";
            const reasons = (o.reasons || []).map((r) => humanizeCleaningReason(r, o)).join("; ");
            return `<li><strong>${o.run_id || "Unknown"}</strong>${meta}: ${reasons}</li>`;
          }).join("")}</ul>
        </div>`
      )
      .join("");
    cleaningHtml = `
      <div class="cleaning-card">
        <h3>Ignored Runs — Pace &amp; Time Cleaning</h3>
        <p>Removed <strong>${cleaning.removed_count}</strong> of ${cleaning.total_before} runs
           (${cleaning.removed_pct}%) with bad pace, finish time, or 3σ outliers.</p>
        ${stageBlocks}
      </div>`;
  } else if (cleaning && !cleaning.skipped) {
    cleaningHtml = `
      <div class="cleaning-card cleaning-ok">
        <p>✓ All ${cleaning.total_before} runs passed pace &amp; time cleaning.</p>
      </div>`;
  }

  runsList.innerHTML = `
    ${cleaningHtml}
    <h3>Runs Used for Learning</h3>
    <div class="runs-table">
      ${profile.runs
        .map(
          (r) => `
        <div class="run-row">
          <span class="run-name">${r.run_id}</span>
          <span>${r.distance_km} km</span>
          <span>${r.avg_pace}</span>
          <span>${r.elevation_gain_m} m elev</span>
          ${r.has_heart_rate ? '<span class="tag">HR</span>' : ""}
          ${r.has_cadence ? '<span class="tag">Cadence</span>' : ""}
        </div>
      `
        )
        .join("")}
    </div>
  `;

  profileSection.classList.remove("hidden");
  recommendForm.classList.remove("hidden");
}

async function learnRuns() {
  hideRecError();
  recLoadingSection.classList.remove("hidden");
  recLoadingText.textContent = "Cleaning outliers & training ML model...";

  try {
    const res = await fetch("/api/runs/learn", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Learning failed");
    renderProfile(data);
  } catch (err) {
    showRecError(err.message);
  } finally {
    recLoadingSection.classList.add("hidden");
  }
}

async function geocodeLocation() {
  const q = locationInput.value.trim();
  if (!q) {
    geocodeResult.textContent = "Enter a location first.";
    return;
  }

  geocodeBtn.disabled = true;
  geocodeResult.textContent = "Searching...";

  try {
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Location not found");
    geocodedLat = data.latitude;
    geocodedLon = data.longitude;
    geocodeResult.textContent = `Found: ${data.name}`;
  } catch (err) {
    geocodedLat = null;
    geocodedLon = null;
    geocodeResult.textContent = err.message;
  } finally {
    geocodeBtn.disabled = false;
  }
}

let lastRecommendationData = null;

function showSelectedRoute(route, isPrimary = false) {
  document.getElementById("rec-title").textContent = route.name || "Your New Route";
  document.getElementById("match-score").textContent = isPrimary
    ? `Top pick · ${(route.scores.total * 100).toFixed(0)}% suit`
    : `Alternative · ${(route.scores.total * 100).toFixed(0)}% suit`;

  const reasoningCard = document.getElementById("reasoning-card");
  if (isPrimary && route.reasoning) {
    reasoningCard.innerHTML = `
      <h3>Why this route?</h3>
      <ul>${route.reasoning.map((r) => `<li>${r}</li>`).join("")}</ul>
    `;
    reasoningCard.classList.remove("hidden");
  } else {
    reasoningCard.innerHTML = `
      <h3>Alternative route</h3>
      <p>Previewing <strong>${route.name}</strong>. Click the top recommendation to switch back.</p>
    `;
    reasoningCard.classList.remove("hidden");
  }

  const items = [
    { label: "Route Type", value: `${route.shape} — ${route.direction}` },
    { label: "Distance", value: route.distance || `${route.distance_km} km` },
    { label: "Elevation Gain", value: `${route.elevation_gain_m} m` },
    { label: "Your Avg Pace", value: route.avg_pace },
    { label: "Suitability", value: `${(route.scores.total * 100).toFixed(0)}%` },
    { label: "Novelty Score", value: `${(route.scores.novelty * 100).toFixed(0)}%` },
  ];
  if (route.predicted_finish_time) {
    items.push({ label: "Estimated Time", value: route.predicted_finish_time });
  }

  document.getElementById("rec-stats-grid").innerHTML = items
    .map(
      (item) => `
      <div class="stat-card">
        <div class="stat-label">${item.label}</div>
        <div class="stat-value">${item.value}</div>
      </div>
    `
    )
    .join("");

  const mapFrame = document.getElementById("rec-pace-map");
  mapFrame.src = "about:blank";
  mapFrame.src = `${route.outputs.pace_map}?t=${Date.now()}`;

  document.querySelectorAll(".alt-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.routeId === route.route_id);
  });
}

function renderRecommendation(data) {
  lastRecommendationData = data;

  const altSection = document.getElementById("alternatives-section");
  const altList = document.getElementById("alternatives-list");

  if (data.alternatives && data.alternatives.length) {
    altList.innerHTML = `
      <div class="alt-card active" data-route-id="${data.recommended.route_id}" data-primary="true" role="button" tabindex="0">
        <span class="alt-name">★ ${data.recommended.name}</span>
        <span>${data.recommended.distance_km} km</span>
        <span>${data.recommended.elevation_gain_m} m elev</span>
        <span class="alt-score">${(data.recommended.scores.total * 100).toFixed(0)}% suit</span>
      </div>
      ${data.alternatives
        .map(
          (alt) => `
        <div class="alt-card" data-route-id="${alt.route_id}" role="button" tabindex="0">
          <span class="alt-name">${alt.name}</span>
          <span>${alt.distance_km} km</span>
          <span>${alt.elevation_gain_m} m elev</span>
          <span class="alt-score">${(alt.scores.total * 100).toFixed(0)}% suit</span>
        </div>
      `
        )
        .join("")}
    `;

    altList.querySelectorAll(".alt-card").forEach((card) => {
      const select = () => {
        const routeId = card.dataset.routeId;
        if (card.dataset.primary === "true") {
          showSelectedRoute(data.recommended, true);
          return;
        }
        const alt = data.alternatives.find((r) => r.route_id === routeId);
        if (alt) showSelectedRoute(alt, false);
      };
      card.addEventListener("click", select);
      card.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });
    });

    altSection.classList.remove("hidden");
  } else {
    altSection.classList.add("hidden");
  }

  showSelectedRoute(data.recommended, true);
  recommendResults.classList.remove("hidden");
}

async function getRecommendation() {
  hideRecError();

  if (geocodedLat === null || geocodedLon === null) {
    showRecError("Find your location first using the 'Find Location' button.");
    return;
  }

  const distance = parseFloat(distanceInput.value);
  if (!distance || distance <= 0) {
    showRecError("Enter a valid target distance.");
    return;
  }

  recLoadingSection.classList.remove("hidden");
  recLoadingText.textContent = "Generating routes (usually 5–10 seconds)...";
  recommendResults.classList.add("hidden");

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: geocodedLat,
        longitude: geocodedLon,
        distance_km: distance,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Recommendation failed");
    renderRecommendation(data);
  } catch (err) {
    showRecError(err.message);
  } finally {
    recLoadingSection.classList.add("hidden");
  }
}

async function loadExistingProfile() {
  try {
    const res = await fetch("/api/runs/profile");
    const data = await res.json();
    if (data.learned) {
      renderProfile(data);
    }
  } catch {
    // Profile not yet available
  }
}

recUploadZone.addEventListener("click", () => recFileInput.click());
recBrowseBtn.addEventListener("click", (e) => { e.stopPropagation(); recFileInput.click(); });
recFileInput.addEventListener("change", () => {
  if (recFileInput.files.length) uploadLibraryFiles(recFileInput.files);
});
recUploadZone.addEventListener("dragover", (e) => { e.preventDefault(); recUploadZone.classList.add("dragover"); });
recUploadZone.addEventListener("dragleave", () => recUploadZone.classList.remove("dragover"));
recUploadZone.addEventListener("drop", (e) => {
  e.preventDefault();
  recUploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) uploadLibraryFiles(e.dataTransfer.files);
});
learnBtn.addEventListener("click", learnRuns);

document.getElementById("reset-library-btn").addEventListener("click", async () => {
  if (!confirm("Clear all uploaded runs, your learned profile, and the trained model?")) {
    return;
  }
  hideRecError();
  recLoadingSection.classList.remove("hidden");
  recLoadingText.textContent = "Resetting library...";
  try {
    const res = await fetch("/api/runs/reset", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Reset failed");
    libraryRunCount = 0;
    updateLibraryStatus();
    profileSection.classList.add("hidden");
    recommendForm.classList.add("hidden");
    recommendResults.classList.add("hidden");
    recFileInput.value = "";
  } catch (err) {
    showRecError(err.message);
  } finally {
    recLoadingSection.classList.add("hidden");
  }
});
geocodeBtn.addEventListener("click", geocodeLocation);
recommendBtn.addEventListener("click", getRecommendation);

// ─── Init ─────────────────────────────────────────────────────────────────────

checkHealth();
loadExistingProfile();

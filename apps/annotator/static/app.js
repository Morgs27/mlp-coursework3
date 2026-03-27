/* ═════════════════════════════════════════════
   Darts Gaze Annotator — Frontend Logic
   ═════════════════════════════════════════════ */

const state = {
  currentVideo: null,
  currentFPS: 30,
  currentROI: null,
  drawingROI: false,
  roiStart: null,
  currentPreviewCaptureId: null,
  knownMatches: window.__ANNOTATOR_BOOTSTRAP__.knownMatches || [],
  allDarts: [],
  liveOverlay: null,
  liveGazeEnabled: false,
  liveGazeBusy: false,
  liveGazeTimer: null,
};

/* ─── Element references ─── */
const $ = (id) => document.getElementById(id);

const el = {
  // top bar
  globalStatus: $("global-status"),
  // source tab
  uploadForm: $("upload-form"),
  videoFile: $("video-file"),
  dropZone: $("drop-zone"),
  dropFileName: $("drop-file-name"),
  uploadBtn: $("upload-btn"),
  youtubeUrl: $("youtube-url"),
  youtubeDownloadBtn: $("youtube-download-btn"),
  youtubeProgress: $("youtube-progress"),
  youtubeProgressFill: $("youtube-progress-fill"),
  youtubeProgressText: $("youtube-progress-text"),
  videoSelect: $("video-select"),
  videoMeta: $("video-meta"),
  videoCount: $("video-count"),
  // match tab
  knownMatchSelect: $("known-match-select"),
  sportEventId: $("sport-event-id"),
  matchDate: $("match-date"),
  matchQuery: $("match-query"),
  matchSearchButton: $("match-search-button"),
  matchSearchResults: $("match-search-results"),
  loadDartsButton: $("load-darts-button"),
  previewMatchButton: $("preview-match-button"),
  timelineStatus: $("timeline-status"),
  candidateList: $("candidate-list"),
  dartCount: $("dart-count"),
  // sync tab
  anchorVideoTime: $("anchor-video-time"),
  anchorEventId: $("anchor-event-id"),
  anchorNotes: $("anchor-notes"),
  saveAnchorButton: $("save-anchor-button"),
  anchorsList: $("anchors-list"),
  anchorCount: $("anchor-count"),
  // viewer
  videoPlayer: $("video-player"),
  overlayCanvas: $("overlay-canvas"),
  currentTime: $("current-time"),
  totalTime: $("total-time"),
  stepBackButton: $("step-back-button"),
  stepForwardButton: $("step-forward-button"),
  skipBack5: $("skip-back-5"),
  skipBack10: $("skip-back-10"),
  skipFwd5: $("skip-fwd-5"),
  skipFwd10: $("skip-fwd-10"),
  playPauseButton: $("play-pause-button"),
  playIcon: $("play-icon"),
  pauseIcon: $("pause-icon"),
  speedSelect: $("speed-select"),
  jumpTimeInput: $("jump-time-input"),
  jumpTimeBtn: $("jump-time-btn"),
  clearROIButton: $("clear-roi-button"),
  annotateNowButton: $("annotate-now-button"),
  liveGazeButton: $("live-gaze-button"),
  gazeRefreshSelect: $("gaze-refresh-select"),
  saveCaptureButton: $("save-capture-button"),
  // right panel
  captureNotes: $("capture-notes"),
  captureStatus: $("capture-status"),
  liveGazeStatus: $("live-gaze-status"),
  capturesList: $("captures-list"),
  captureCount: $("capture-count"),
  capturePreviewModal: $("capture-preview-modal"),
  closeCapturePreview: $("close-capture-preview"),
  capturePreviewTitle: $("capture-preview-title"),
  capturePreviewImage: $("capture-preview-image"),
};

const overlayCtx = el.overlayCanvas.getContext("2d");

/* ─── Helpers ─── */
async function fetchJSON(url, options = {}) {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `Request failed (${resp.status})`);
  return data;
}

function currentSportEventId() {
  return el.sportEventId.value.trim();
}

function requireVideo() {
  if (!state.currentVideo) throw new Error("Select a video first.");
}

function fmt(s) {
  const val = Number(s || 0);
  const mins = Math.floor(val / 60);
  const secs = (val % 60).toFixed(3);
  return `${mins}:${secs.padStart(6, "0")}`;
}

function fmtShort(s) {
  return Number(s || 0).toFixed(3);
}

function setStatus(element, msg, cls = "") {
  element.textContent = msg;
  element.className = `status-text ${cls}`.trim();
}

function setGlobalStatus(msg, type = "") {
  el.globalStatus.textContent = msg;
  el.globalStatus.className = `status-pill ${type}`.trim();
  // auto-clear after a while
  if (msg) setTimeout(() => {
    if (el.globalStatus.textContent === msg) {
      el.globalStatus.textContent = "";
      el.globalStatus.className = "status-pill";
    }
  }, 8000);
}

/* ─── Tab switching ─── */
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(btn.dataset.tab).classList.add("active");
  });
});

/* ═══ VIDEO OVERLAY / ROI ═══ */
function syncCanvas() {
  const rect = el.videoPlayer.getBoundingClientRect();
  el.overlayCanvas.width = rect.width;
  el.overlayCanvas.height = rect.height;
  // position the canvas exactly over the <video>
  el.overlayCanvas.style.width = rect.width + "px";
  el.overlayCanvas.style.height = rect.height + "px";
  el.overlayCanvas.style.left = (el.videoPlayer.offsetLeft) + "px";
  el.overlayCanvas.style.top = (el.videoPlayer.offsetTop) + "px";
  drawROI();
}

function projectVideoPoint(point) {
  if (!state.currentVideo) return { x: point.x, y: point.y };
  return {
    x: point.x * (el.overlayCanvas.width / state.currentVideo.frame_width),
    y: point.y * (el.overlayCanvas.height / state.currentVideo.frame_height),
  };
}

function drawArrow(start, end, color, lineWidth = 2) {
  overlayCtx.save();
  overlayCtx.strokeStyle = color;
  overlayCtx.fillStyle = color;
  overlayCtx.lineWidth = lineWidth;
  overlayCtx.lineCap = "round";
  overlayCtx.beginPath();
  overlayCtx.moveTo(start.x, start.y);
  overlayCtx.lineTo(end.x, end.y);
  overlayCtx.stroke();
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const headLength = 8;
  overlayCtx.beginPath();
  overlayCtx.moveTo(end.x, end.y);
  overlayCtx.lineTo(end.x - headLength * Math.cos(angle - Math.PI / 6), end.y - headLength * Math.sin(angle - Math.PI / 6));
  overlayCtx.lineTo(end.x - headLength * Math.cos(angle + Math.PI / 6), end.y - headLength * Math.sin(angle + Math.PI / 6));
  overlayCtx.closePath();
  overlayCtx.fill();
  overlayCtx.restore();
}

function drawLiveOverlay() {
  if (!state.liveOverlay || !state.currentVideo) return;
  const overlay = state.liveOverlay;

  if (overlay.face_bbox) {
    const topLeft = projectVideoPoint({ x: overlay.face_bbox.x, y: overlay.face_bbox.y });
    const bottomRight = projectVideoPoint({
      x: overlay.face_bbox.x + overlay.face_bbox.width,
      y: overlay.face_bbox.y + overlay.face_bbox.height,
    });
    overlayCtx.save();
    overlayCtx.strokeStyle = "#2a9d8f";
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([8, 5]);
    overlayCtx.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    overlayCtx.setLineDash([]);
    overlayCtx.restore();
  }

  const colorMap = {
    left: "#58a6ff",
    right: "#f47067",
    average: "#d29922",
    x: "#ff6b6b",
    y: "#3fb950",
    z: "#79c0ff",
  };

  (overlay.gaze_arrows || []).forEach((arrow) => {
    drawArrow(projectVideoPoint(arrow.start), projectVideoPoint(arrow.end), colorMap[arrow.label] || "#ffffff", arrow.label === "average" ? 3 : 2);
  });
  (overlay.head_axes || []).forEach((axis) => {
    drawArrow(projectVideoPoint(axis.start), projectVideoPoint(axis.end), colorMap[axis.label] || "#ffffff", 2);
  });
}

function drawROI() {
  overlayCtx.clearRect(0, 0, el.overlayCanvas.width, el.overlayCanvas.height);
  drawLiveOverlay();
  if (!state.currentROI) return;
  overlayCtx.strokeStyle = "#f47067";
  overlayCtx.lineWidth = 2;
  overlayCtx.setLineDash([6, 4]);
  overlayCtx.strokeRect(state.currentROI.x, state.currentROI.y, state.currentROI.width, state.currentROI.height);
  overlayCtx.setLineDash([]);

  // semi-transparent fill
  overlayCtx.fillStyle = "rgba(244, 112, 103, 0.08)";
  overlayCtx.fillRect(state.currentROI.x, state.currentROI.y, state.currentROI.width, state.currentROI.height);
}

function canvasROIToVideoROI() {
  if (!state.currentROI || !state.currentVideo) return null;
  const sx = state.currentVideo.frame_width / el.overlayCanvas.width;
  const sy = state.currentVideo.frame_height / el.overlayCanvas.height;
  return {
    x: Math.round(state.currentROI.x * sx),
    y: Math.round(state.currentROI.y * sy),
    width: Math.round(state.currentROI.width * sx),
    height: Math.round(state.currentROI.height * sy),
  };
}

function openCapturePreview(title, src) {
  el.capturePreviewTitle.textContent = title;
  el.capturePreviewImage.src = `${src}${src.includes("?") ? "&" : "?"}t=${Date.now()}`;
  el.capturePreviewModal.classList.remove("hidden");
  el.capturePreviewModal.setAttribute("aria-hidden", "false");
}

function closeCapturePreview() {
  el.capturePreviewModal.classList.add("hidden");
  el.capturePreviewModal.setAttribute("aria-hidden", "true");
  el.capturePreviewImage.removeAttribute("src");
}

/* ─── Time display update ─── */
function updateTimeDisplay() {
  el.currentTime.textContent = fmt(el.videoPlayer.currentTime);
  el.totalTime.textContent = fmt(el.videoPlayer.duration || 0);
}

/* ═══ VIDEO MANAGEMENT ═══ */
function renderVideos(videos) {
  el.videoSelect.innerHTML = "";
  el.videoCount.textContent = videos.length;
  if (!videos.length) {
    el.videoSelect.innerHTML = "<option value=''>No videos yet</option>";
    return;
  }
  videos.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v.id;
    opt.textContent = `${v.display_name} (${fmtShort(v.duration_s)}s)`;
    el.videoSelect.appendChild(opt);
  });
  if (!state.currentVideo) selectVideo(videos[0]);
}

async function loadVideos() {
  const videos = await fetchJSON("/api/videos", { headers: {} });
  renderVideos(videos);
}

function selectVideo(video) {
  stopLiveGaze({ clearOverlay: true, clearStatus: true });
  state.currentVideo = video;
  state.currentFPS = video.fps || 30;
  el.videoSelect.value = video.id;
  el.videoPlayer.src = `/media/videos/${video.id}`;

  // Render metadata grid
  el.videoMeta.innerHTML = `
    <div class="meta-item"><span class="meta-label">FPS</span><span class="meta-value">${fmtShort(video.fps)}</span></div>
    <div class="meta-item"><span class="meta-label">Duration</span><span class="meta-value">${fmtShort(video.duration_s)}s</span></div>
    <div class="meta-item"><span class="meta-label">Resolution</span><span class="meta-value">${video.frame_width}×${video.frame_height}</span></div>
    <div class="meta-item"><span class="meta-label">Source</span><span class="meta-value">${video.source_url ? "YouTube" : "Upload"}</span></div>
  `;

  state.currentROI = null;
  state.liveOverlay = null;
  drawROI();
  loadAnchors();
  loadCaptures();
  setGlobalStatus(`Loaded: ${video.display_name}`, "ok");
}

async function refreshSelectedVideo() {
  const videos = await fetchJSON("/api/videos", { headers: {} });
  renderVideos(videos);
  const sel = videos.find((v) => String(v.id) === el.videoSelect.value) || videos[0];
  if (sel) selectVideo(sel);
}

/* ═══ MATCH SEARCH ═══ */
function renderMatches(matches) {
  el.matchSearchResults.innerHTML = "";
  if (!matches.length) {
    el.matchSearchResults.innerHTML = '<div class="helper-text" style="padding:8px">No matches found.</div>';
    return;
  }
  matches.forEach((m) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <strong>${m.title}</strong>
      <div>${m.start_time || ""}</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--text-dim)">${m.sport_event_id}</div>
      <div class="item-actions">
        <button type="button" class="btn btn-use btn-sm">Use Match</button>
      </div>`;
    item.querySelector(".btn-use").addEventListener("click", () => {
      el.sportEventId.value = m.sport_event_id;
      setGlobalStatus(`Selected: ${m.title}`, "ok");
    });
    el.matchSearchResults.appendChild(item);
  });
}

/* ═══ ANCHORS ═══ */
function renderAnchors(anchors) {
  el.anchorsList.innerHTML = "";
  el.anchorCount.textContent = anchors.length;
  anchors.forEach((a) => {
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <strong>${fmtShort(a.video_time_s)}s → event ${a.timeline_event_id}</strong>
      <div>${a.notes || ""}</div>
      <div class="item-actions">
        <button type="button" class="btn btn-danger btn-sm">Delete</button>
      </div>`;
    item.querySelector(".btn-danger").addEventListener("click", async () => {
      await fetchJSON(`/api/anchors/${a.id}`, { method: "DELETE", headers: {} });
      await loadAnchors();
    });
    el.anchorsList.appendChild(item);
  });
}

async function loadAnchors() {
  if (!state.currentVideo || !currentSportEventId()) {
    el.anchorsList.innerHTML = "";
    el.anchorCount.textContent = "0";
    return;
  }
  const anchors = await fetchJSON(`/api/anchors?video_id=${state.currentVideo.id}&sport_event_id=${encodeURIComponent(currentSportEventId())}`, { headers: {} });
  renderAnchors(anchors);
}

/* ═══ CAPTURES ═══ */
function renderCaptures(captures) {
  el.capturesList.innerHTML = "";
  el.captureCount.textContent = captures.length;

  if (!captures.length) {
    el.capturesList.innerHTML = '<div class="helper-text" style="padding:12px;text-align:center">No captures yet.<br>Use the Capture button or press <strong>C</strong>.</div>';
    return;
  }

  captures.forEach((c) => {
    const card = document.createElement("div");
    card.className = "capture-card";
    const statusCls = c.review_status.replace(/\s+/g, "_");

    // Build event info line
    let eventLine;
    if (c.matched_throw_event_id) {
      // Try to find the throw in the loaded darts if available
      const throwInfo = state.allDarts.find((d) => d.throw_event_id === c.matched_throw_event_id);
      if (throwInfo) {
        eventLine = `<div class="capture-event"><strong>${throwInfo.segment_label} · ${throwInfo.resulting_score}</strong> ${throwInfo.player_name}</div>`;
      } else {
        eventLine = `<div class="capture-event"><strong>Throw ${c.matched_throw_event_id}</strong></div>`;
      }
    } else {
      eventLine = `<div class="capture-event unmatched"><strong>Unmatched</strong></div>`;
    }

    // Notes line
    const notesLine = c.notes ? `<div class="capture-notes-text">${c.notes}</div>` : "";

    // Resolved time
    const resolvedLine = c.resolved_timeline_time_utc
      ? `<div class="capture-time">${c.resolved_timeline_time_utc.replace("T", " ").slice(0, 19)}</div>`
      : "";

    card.innerHTML = `
      <div class="capture-card-inner">
        <div class="capture-thumb" title="Click to jump to ${fmtShort(c.video_time_s)}s">
          <img src="/media/captures/${c.id}" alt="Capture ${c.id}" loading="lazy">
        </div>
        <div class="capture-details">
          <span class="capture-id">#${c.id}</span>
          <div class="capture-time">${fmt(c.video_time_s)}</div>
          ${eventLine}
          ${resolvedLine}
          ${notesLine}
        </div>
      </div>
      <div class="capture-footer">
        <div class="capture-footer-top">
          <span class="capture-status-badge ${statusCls}">${c.review_status}</span>
          <div class="capture-actions">
            <button type="button" class="btn-goto-capture" title="Jump video to this time">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
              Go to
            </button>
            <button type="button" class="btn-delete-capture" title="Delete this capture">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
              Delete
            </button>
          </div>
        </div>
        <div class="capture-footer-bottom">
          <button type="button" class="btn-preview-capture" title="View raw frame">Raw</button>
          <button type="button" class="btn-preview-capture" title="View annotated frame">Annotated</button>
          ${c.face_bbox ? '<button type="button" class="btn-preview-capture btn-preview-roi" title="View annotated ROI">ROI</button>' : ""}
        </div>
      </div>`;

    // Preview raw capture on thumbnail click
    card.querySelector(".capture-thumb").addEventListener("click", () => {
      openCapturePreview(`Capture #${c.id} · Raw`, c.media_urls.raw);
    });

    card.querySelector(".btn-preview-capture").addEventListener("click", () => {
      openCapturePreview(`Capture #${c.id} · Raw`, c.media_urls.raw);
    });
    card.querySelectorAll(".btn-preview-capture")[1].addEventListener("click", () => {
      openCapturePreview(`Capture #${c.id} · Annotated`, c.media_urls.annotated);
    });
    const roiButton = card.querySelector(".btn-preview-roi");
    if (roiButton) {
      roiButton.addEventListener("click", () => {
        openCapturePreview(`Capture #${c.id} · ROI`, c.media_urls.annotated_roi);
      });
    }

    // Go to button
    card.querySelector(".btn-goto-capture").addEventListener("click", () => {
      el.videoPlayer.currentTime = c.video_time_s;
    });

    // Delete button
    card.querySelector(".btn-delete-capture").addEventListener("click", async () => {
      if (!confirm(`Delete capture #${c.id}?`)) return;
      try {
        await fetchJSON(`/api/captures/${c.id}`, { method: "DELETE", headers: {} });
        setGlobalStatus(`Deleted capture #${c.id}`, "ok");
        await loadCaptures();
      } catch (err) {
        setGlobalStatus(`Delete failed: ${err.message}`, "warn");
      }
    })

    el.capturesList.appendChild(card);
  });
}

async function loadCaptures() {
  if (!state.currentVideo) {
    el.capturesList.innerHTML = "";
    el.captureCount.textContent = "0";
    return;
  }
  const q = new URLSearchParams({ video_id: String(state.currentVideo.id) });
  if (currentSportEventId()) q.set("sport_event_id", currentSportEventId());
  const captures = await fetchJSON(`/api/captures?${q}`, { headers: {} });
  renderCaptures(captures);
}

/* ═══ TIMELINE / CANDIDATES ═══ */
function renderCandidates(candidates, captureId = null) {
  state.currentPreviewCaptureId = captureId;
  el.candidateList.innerHTML = "";
  if (!candidates.length) {
    el.candidateList.innerHTML = '<div class="helper-text" style="padding:8px">No candidates.</div>';
    return;
  }
  let lastPeriod = null;
  candidates.forEach((c) => {
    if (c.period !== undefined && c.period !== null && c.period !== lastPeriod) {
      lastPeriod = c.period;
      const header = document.createElement("div");
      header.className = "period-header";
      header.innerHTML = `Set ${c.period}`;
      el.candidateList.appendChild(header);
    }
    const item = document.createElement("div");
    item.className = "list-item";
    item.innerHTML = `
      <strong>${c.segment_label} · ${c.resulting_score}</strong>
      <div>${c.player_name} · Dart ${c.dart_in_visit}</div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--text-dim)">${c.throw_time_utc} · event ${c.throw_event_id}</div>`;
    if (captureId) {
      const actions = document.createElement("div");
      actions.className = "item-actions";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn btn-use btn-sm";
      btn.textContent = "Use as match";
      btn.addEventListener("click", async () => {
        await fetchJSON(`/api/captures/${captureId}`, {
          method: "PATCH",
          body: JSON.stringify({ matched_throw_event_id: c.throw_event_id, review_status: "verified" }),
        });
        setGlobalStatus(`Capture ${captureId} → throw ${c.throw_event_id}`, "ok");
        await loadCaptures();
      });
      actions.appendChild(btn);
      item.appendChild(actions);
    }
    el.candidateList.appendChild(item);
  });
}

async function loadDartTimeline() {
  const eid = currentSportEventId();
  if (!eid) throw new Error("Enter a sport_event_id first.");
  const darts = await fetchJSON(`/api/events/${encodeURIComponent(eid)}/darts`, { headers: {} });
  state.allDarts = darts;
  el.dartCount.textContent = darts.length;
  setStatus(el.timelineStatus, `Loaded ${darts.length} dart events`, "status-ok");
  renderCandidates(darts);
}

async function previewMatchResolution() {
  requireVideo();
  const eid = currentSportEventId();
  if (!eid) throw new Error("Enter a sport_event_id first.");
  const q = new URLSearchParams({
    video_id: String(state.currentVideo.id),
    sport_event_id: eid,
    video_time_s: String(el.videoPlayer.currentTime),
  });
  const res = await fetchJSON(`/api/match-resolution?${q}`, { headers: {} });
  const lines = [
    `Mapped: ${res.mapped_time_utc || "—"}`,
    `Status: ${res.resolution_status}`,
    `Suggestion: ${res.matched_throw_event_id || "none"}`,
  ].join("\n");
  setStatus(el.timelineStatus, lines, res.ambiguous ? "status-review" : "status-ok");
  renderCandidates(res.candidates || []);
}

/* ═══ CAPTURE ═══ */
function captureFrameDataURL() {
  const canvas = document.createElement("canvas");
  canvas.width = state.currentVideo.frame_width;
  canvas.height = state.currentVideo.frame_height;
  canvas.getContext("2d").drawImage(el.videoPlayer, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/png");
}

async function saveCapture() {
  requireVideo();
  const payload = {
    video_id: state.currentVideo.id,
    sport_event_id: currentSportEventId() || null,
    video_time_s: el.videoPlayer.currentTime,
    image_data: captureFrameDataURL(),
    face_bbox: canvasROIToVideoROI(),
    notes: el.captureNotes.value.trim() || null,
  };
  const resp = await fetchJSON("/api/captures", { method: "POST", body: JSON.stringify(payload) });
  const cap = resp.capture;
  const res = resp.resolution || {};
  const lines = [`Saved capture #${cap.id} at ${fmtShort(cap.video_time_s)}s`, `Status: ${cap.review_status}`];
  if (res.mapped_time_utc) lines.push(`Mapped: ${res.mapped_time_utc}`);
  if (res.matched_throw_event_id) lines.push(`Throw: ${res.matched_throw_event_id}`);
  setStatus(el.captureStatus, lines.join("\n"), res.ambiguous ? "status-review" : "status-ok");
  setGlobalStatus(`Captured frame #${cap.id}`, "ok");
  renderCandidates(res.candidates || [], cap.id);
  await loadCaptures();
}

async function requestCurrentFrameAnnotation() {
  requireVideo();
  if (state.liveGazeBusy || el.videoPlayer.readyState < 2) return null;
  state.liveGazeBusy = true;
  try {
    const response = await fetchJSON("/api/gaze/annotate-frame", {
      method: "POST",
      body: JSON.stringify({
        image_data: captureFrameDataURL(),
        face_bbox: canvasROIToVideoROI(),
      }),
    });
    state.liveOverlay = response.overlay || null;
    drawROI();
    setStatus(
      el.liveGazeStatus,
      response.valid_face ? "Live gaze overlay updated." : "Live gaze: no face detected in the current frame.",
      response.valid_face ? "status-ok" : "status-review",
    );
    return response;
  } catch (error) {
    setStatus(el.liveGazeStatus, `Live gaze failed: ${error.message}`, "status-review");
    throw error;
  } finally {
    state.liveGazeBusy = false;
  }
}

function updateLiveGazeButton() {
  if (state.liveGazeEnabled) {
    el.liveGazeButton.classList.add("live-active");
    el.liveGazeButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12"/></svg>
      Stop Live
    `;
  } else {
    el.liveGazeButton.classList.remove("live-active");
    el.liveGazeButton.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Live Gaze
    `;
  }
}

function stopLiveGaze({ clearOverlay = false, clearStatus = false } = {}) {
  if (state.liveGazeTimer) {
    clearInterval(state.liveGazeTimer);
    state.liveGazeTimer = null;
  }
  state.liveGazeEnabled = false;
  if (clearOverlay) {
    state.liveOverlay = null;
    drawROI();
  }
  if (clearStatus) {
    setStatus(el.liveGazeStatus, "", "");
  }
  updateLiveGazeButton();
}

async function startLiveGaze() {
  requireVideo();
  stopLiveGaze();
  state.liveGazeEnabled = true;
  updateLiveGazeButton();
  await requestCurrentFrameAnnotation();
  const intervalMs = Math.max(120, Math.round(1000 / Number(el.gazeRefreshSelect.value || 4)));
  state.liveGazeTimer = setInterval(() => {
    if (!state.liveGazeEnabled || state.liveGazeBusy) return;
    if (el.videoPlayer.paused || el.videoPlayer.ended) return;
    requestCurrentFrameAnnotation().catch(() => { });
  }, intervalMs);
}

/* ═══ YOUTUBE DOWNLOAD ═══ */
async function startYoutubeDownload() {
  const url = el.youtubeUrl.value.trim();
  if (!url) return;
  el.youtubeProgress.hidden = false;
  el.youtubeDownloadBtn.disabled = true;
  el.youtubeProgressFill.style.width = "0%";
  el.youtubeProgressText.textContent = "Starting download…";

  try {
    const { job_id } = await fetchJSON("/api/videos/youtube", {
      method: "POST",
      body: JSON.stringify({ url }),
    });

    const poll = setInterval(async () => {
      try {
        const job = await fetchJSON(`/api/videos/youtube/status/${job_id}`, { headers: {} });
        el.youtubeProgressFill.style.width = `${job.progress}%`;

        if (job.status === "downloading") {
          el.youtubeProgressText.textContent = `Downloading… ${job.progress}%`;
        } else if (job.status === "done") {
          clearInterval(poll);
          el.youtubeProgressText.textContent = "Done!";
          el.youtubeDownloadBtn.disabled = false;
          setGlobalStatus("YouTube video downloaded", "ok");
          await refreshSelectedVideo();
          setTimeout(() => { el.youtubeProgress.hidden = true; }, 2000);
        } else if (job.status === "error") {
          clearInterval(poll);
          el.youtubeProgressText.textContent = `Error: ${job.error}`;
          el.youtubeDownloadBtn.disabled = false;
          setGlobalStatus("YouTube download failed", "warn");
        }
      } catch (e) {
        clearInterval(poll);
        el.youtubeProgressText.textContent = `Poll error: ${e.message}`;
        el.youtubeDownloadBtn.disabled = false;
      }
    }, 1500);
  } catch (e) {
    el.youtubeProgressText.textContent = e.message;
    el.youtubeDownloadBtn.disabled = false;
    setGlobalStatus("YouTube download failed", "warn");
  }
}

/* ═══════════════════════════════════════
   EVENT WIRING
   ═══════════════════════════════════════ */

/* ─ File upload ─ */
el.videoFile.addEventListener("change", () => {
  const file = el.videoFile.files[0];
  el.dropFileName.textContent = file ? file.name : "";
  el.uploadBtn.disabled = !file;
});

// drag & drop
el.dropZone.addEventListener("dragover", (e) => { e.preventDefault(); el.dropZone.classList.add("drag-over"); });
el.dropZone.addEventListener("dragleave", () => el.dropZone.classList.remove("drag-over"));
el.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  el.dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    el.videoFile.files = e.dataTransfer.files;
    el.dropFileName.textContent = e.dataTransfer.files[0].name;
    el.uploadBtn.disabled = false;
  }
});

el.uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!el.videoFile.files.length) return;
  el.uploadBtn.disabled = true;
  setGlobalStatus("Uploading…");
  const fd = new FormData();
  fd.append("video", el.videoFile.files[0]);
  try {
    const resp = await fetch("/api/videos/upload", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) { setGlobalStatus(data.error || "Upload failed", "warn"); return; }
    await refreshSelectedVideo();
    setGlobalStatus(`Uploaded: ${data.display_name}`, "ok");
    el.dropFileName.textContent = "";
  } catch (err) {
    setGlobalStatus(err.message, "warn");
  } finally {
    el.uploadBtn.disabled = false;
  }
});

/* ─ YouTube download ─ */
el.youtubeDownloadBtn.addEventListener("click", startYoutubeDownload);

/* ─ Video selection ─ */
el.videoSelect.addEventListener("change", async () => {
  const videos = await fetchJSON("/api/videos", { headers: {} });
  const sel = videos.find((v) => String(v.id) === el.videoSelect.value);
  if (sel) selectVideo(sel);
});

/* ─ Known match ─ */
el.knownMatchSelect.addEventListener("change", () => {
  const opt = el.knownMatchSelect.selectedOptions[0];
  if (opt && opt.value) {
    el.sportEventId.value = opt.value;
    // Auto-fill YouTube URL if available
    const ytUrl = opt.dataset.youtube;
    if (ytUrl) el.youtubeUrl.value = ytUrl;
    loadAnchors().catch((e) => setStatus(el.timelineStatus, e.message, "status-review"));
    loadCaptures().catch((e) => setStatus(el.captureStatus, e.message, "status-review"));
    setGlobalStatus(`Match: ${opt.textContent}`, "ok");
  }
});

/* ─ Match search ─ */
el.matchSearchButton.addEventListener("click", async () => {
  const params = new URLSearchParams();
  if (el.matchDate.value) params.set("date", el.matchDate.value);
  if (el.matchQuery.value.trim()) params.set("query", el.matchQuery.value.trim());
  const matches = await fetchJSON(`/api/matches/search?${params}`, { headers: {} });
  renderMatches(matches);
});

/* ─ Timeline ─ */
el.loadDartsButton.addEventListener("click", () => {
  loadDartTimeline().catch((e) => setStatus(el.timelineStatus, e.message, "status-review"));
});

el.previewMatchButton.addEventListener("click", () => {
  previewMatchResolution().catch((e) => setStatus(el.timelineStatus, e.message, "status-review"));
});

/* ─ Anchors ─ */
el.saveAnchorButton.addEventListener("click", async () => {
  requireVideo();
  const payload = {
    video_id: state.currentVideo.id,
    sport_event_id: currentSportEventId(),
    video_time_s: Number(el.anchorVideoTime.value || el.videoPlayer.currentTime),
    timeline_event_id: Number(el.anchorEventId.value),
    notes: el.anchorNotes.value.trim() || null,
  };
  await fetchJSON("/api/anchors", { method: "POST", body: JSON.stringify(payload) });
  setGlobalStatus("Anchor saved", "ok");
  await loadAnchors();
});

/* ─── Playback helpers ─── */
function skipTime(delta) {
  const dur = el.videoPlayer.duration || Infinity;
  el.videoPlayer.currentTime = Math.max(0, Math.min(dur, el.videoPlayer.currentTime + delta));
}

function togglePlayPause() {
  if (el.videoPlayer.paused) {
    el.videoPlayer.play();
  } else {
    el.videoPlayer.pause();
  }
}

function syncPlayPauseIcon() {
  const paused = el.videoPlayer.paused;
  el.playIcon.style.display = paused ? "block" : "none";
  el.pauseIcon.style.display = paused ? "none" : "block";
}

function parseJumpTime(input) {
  const s = input.trim();
  // "m:ss" or "m:ss.ms" format
  const parts = s.split(":");
  if (parts.length === 2) {
    return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
  }
  // plain seconds
  return parseFloat(s);
}

/* ─ Frame stepping ─ */
el.stepBackButton.addEventListener("click", () => skipTime(-1 / state.currentFPS));
el.stepForwardButton.addEventListener("click", () => skipTime(1 / state.currentFPS));

/* ─ Skip buttons ─ */
el.skipBack5.addEventListener("click", () => skipTime(-5));
el.skipBack10.addEventListener("click", () => skipTime(-10));
el.skipFwd5.addEventListener("click", () => skipTime(5));
el.skipFwd10.addEventListener("click", () => skipTime(10));

/* ─ Play / Pause ─ */
el.playPauseButton.addEventListener("click", togglePlayPause);
el.videoPlayer.addEventListener("play", () => {
  syncPlayPauseIcon();
  if (state.liveGazeEnabled) {
    requestCurrentFrameAnnotation().catch(() => { });
  }
});
el.videoPlayer.addEventListener("pause", () => {
  syncPlayPauseIcon();
  if (state.liveGazeEnabled) {
    requestCurrentFrameAnnotation().catch(() => { });
  }
});
el.videoPlayer.addEventListener("seeked", () => {
  if (state.liveGazeEnabled) {
    requestCurrentFrameAnnotation().catch(() => { });
  }
});

/* ─ Speed ─ */
el.speedSelect.addEventListener("change", () => {
  el.videoPlayer.playbackRate = parseFloat(el.speedSelect.value);
});

/* ─ Jump to time ─ */
function doJump() {
  const t = parseJumpTime(el.jumpTimeInput.value);
  if (!isNaN(t)) {
    el.videoPlayer.currentTime = Math.max(0, Math.min(el.videoPlayer.duration || Infinity, t));
    el.jumpTimeInput.value = "";
  }
}
el.jumpTimeBtn.addEventListener("click", doJump);
el.jumpTimeInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); doJump(); }
});

/* ─ ROI drawing ─ */
el.clearROIButton.addEventListener("click", () => {
  state.currentROI = null;
  drawROI();
});

el.annotateNowButton.addEventListener("click", () => {
  requestCurrentFrameAnnotation().catch((e) => setStatus(el.liveGazeStatus, e.message, "status-review"));
});

el.liveGazeButton.addEventListener("click", () => {
  if (state.liveGazeEnabled) {
    stopLiveGaze({ clearOverlay: true, clearStatus: true });
    setGlobalStatus("Live gaze stopped", "ok");
    return;
  }
  startLiveGaze()
    .then(() => setGlobalStatus("Live gaze started", "ok"))
    .catch((e) => setStatus(el.liveGazeStatus, e.message, "status-review"));
});

el.gazeRefreshSelect.addEventListener("change", () => {
  if (!state.liveGazeEnabled) return;
  startLiveGaze().catch((e) => setStatus(el.liveGazeStatus, e.message, "status-review"));
});

el.overlayCanvas.addEventListener("pointerdown", (e) => {
  state.drawingROI = true;
  const rect = el.overlayCanvas.getBoundingClientRect();
  state.roiStart = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  state.currentROI = { ...state.roiStart, width: 0, height: 0 };
  drawROI();
});

el.overlayCanvas.addEventListener("pointermove", (e) => {
  if (!state.drawingROI || !state.roiStart) return;
  const rect = el.overlayCanvas.getBoundingClientRect();
  const cur = { x: e.clientX - rect.left, y: e.clientY - rect.top };
  state.currentROI = {
    x: Math.min(state.roiStart.x, cur.x),
    y: Math.min(state.roiStart.y, cur.y),
    width: Math.abs(cur.x - state.roiStart.x),
    height: Math.abs(cur.y - state.roiStart.y),
  };
  drawROI();
});

["pointerup", "pointerleave"].forEach((evt) => {
  el.overlayCanvas.addEventListener(evt, () => {
    state.drawingROI = false;
    state.roiStart = null;
    drawROI();
  });
});

/* ─ Capture ─ */
el.saveCaptureButton.addEventListener("click", () => {
  saveCapture().catch((e) => setStatus(el.captureStatus, e.message, "status-review"));
});

/* ─ Video events ─ */
el.videoPlayer.addEventListener("loadedmetadata", () => {
  syncCanvas();
  updateTimeDisplay();
  syncPlayPauseIcon();
});
el.videoPlayer.addEventListener("timeupdate", updateTimeDisplay);
window.addEventListener("resize", syncCanvas);

/* ─ Modal ─ */
el.closeCapturePreview.addEventListener("click", closeCapturePreview);
document.querySelectorAll("[data-close-modal]").forEach((node) => {
  node.addEventListener("click", closeCapturePreview);
});

/* ─── Keyboard shortcuts ─── */
document.addEventListener("keydown", (e) => {
  // Don't intercept when typing in inputs
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;

  switch (e.key) {
    case "ArrowLeft":
      e.preventDefault();
      skipTime(e.shiftKey ? -5 : -1 / state.currentFPS);
      break;
    case "ArrowRight":
      e.preventDefault();
      skipTime(e.shiftKey ? 5 : 1 / state.currentFPS);
      break;
    case " ":
      e.preventDefault();
      togglePlayPause();
      break;
    case "c":
      saveCapture().catch((e2) => setStatus(el.captureStatus, e2.message, "status-review"));
      break;
    case "Escape":
      if (!el.capturePreviewModal.classList.contains("hidden")) {
        closeCapturePreview();
        break;
      }
      state.currentROI = null;
      drawROI();
      break;
  }
});

/* ═══ INIT ═══ */
updateLiveGazeButton();
loadVideos().catch((e) => setGlobalStatus(e.message, "warn"));

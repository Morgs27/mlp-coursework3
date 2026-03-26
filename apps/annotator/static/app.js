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
};

/* ─── Element references ─── */
const $ = (id) => document.getElementById(id);

const el = {
  // top bar
  globalStatus:     $("global-status"),
  // source tab
  uploadForm:       $("upload-form"),
  videoFile:        $("video-file"),
  dropZone:         $("drop-zone"),
  dropFileName:     $("drop-file-name"),
  uploadBtn:        $("upload-btn"),
  youtubeUrl:       $("youtube-url"),
  youtubeDownloadBtn: $("youtube-download-btn"),
  youtubeProgress:  $("youtube-progress"),
  youtubeProgressFill: $("youtube-progress-fill"),
  youtubeProgressText: $("youtube-progress-text"),
  videoSelect:      $("video-select"),
  videoMeta:        $("video-meta"),
  videoCount:       $("video-count"),
  // match tab
  knownMatchSelect: $("known-match-select"),
  sportEventId:     $("sport-event-id"),
  matchDate:        $("match-date"),
  matchQuery:       $("match-query"),
  matchSearchButton: $("match-search-button"),
  matchSearchResults: $("match-search-results"),
  loadDartsButton:  $("load-darts-button"),
  previewMatchButton: $("preview-match-button"),
  timelineStatus:   $("timeline-status"),
  candidateList:    $("candidate-list"),
  dartCount:        $("dart-count"),
  // sync tab
  anchorVideoTime:  $("anchor-video-time"),
  anchorEventId:    $("anchor-event-id"),
  anchorNotes:      $("anchor-notes"),
  saveAnchorButton: $("save-anchor-button"),
  anchorsList:      $("anchors-list"),
  anchorCount:      $("anchor-count"),
  // viewer
  videoPlayer:      $("video-player"),
  overlayCanvas:    $("overlay-canvas"),
  currentTime:      $("current-time"),
  totalTime:        $("total-time"),
  stepBackButton:   $("step-back-button"),
  stepForwardButton: $("step-forward-button"),
  skipBack5:        $("skip-back-5"),
  skipBack10:       $("skip-back-10"),
  skipFwd5:         $("skip-fwd-5"),
  skipFwd10:        $("skip-fwd-10"),
  playPauseButton:  $("play-pause-button"),
  playIcon:         $("play-icon"),
  pauseIcon:        $("pause-icon"),
  speedSelect:      $("speed-select"),
  jumpTimeInput:    $("jump-time-input"),
  jumpTimeBtn:      $("jump-time-btn"),
  clearROIButton:   $("clear-roi-button"),
  saveCaptureButton: $("save-capture-button"),
  // right panel
  captureNotes:     $("capture-notes"),
  captureStatus:    $("capture-status"),
  capturesList:     $("captures-list"),
  captureCount:     $("capture-count"),
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

function drawROI() {
  overlayCtx.clearRect(0, 0, el.overlayCanvas.width, el.overlayCanvas.height);
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
      </div>`;

    // Jump to time on thumbnail click
    card.querySelector(".capture-thumb").addEventListener("click", () => {
      el.videoPlayer.currentTime = c.video_time_s;
    });

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
    });

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
  candidates.forEach((c) => {
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
  renderCandidates(darts.slice(0, 50));
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
el.videoPlayer.addEventListener("play", syncPlayPauseIcon);
el.videoPlayer.addEventListener("pause", syncPlayPauseIcon);

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
      state.currentROI = null;
      drawROI();
      break;
  }
});

/* ═══ INIT ═══ */
loadVideos().catch((e) => setGlobalStatus(e.message, "warn"));

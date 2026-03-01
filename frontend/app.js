const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const ACCEPTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const PROCESSING_TIMEOUT_MS = 90 * 60 * 1000;
const POLL_INTERVAL_MS = 1500;
const STALL_NO_ACTIVITY_MS = 5 * 60 * 1000;
const STATUS_FETCH_MAX_DOWNTIME_MS = 3 * 60 * 1000;
const NO_EVENT_HEARTBEAT_MS = 30 * 1000;
const APP_BUILD = "20260301c";

const PRESETS = {
  safe: { resolution: 384, steps: 16, cfg: 2.5 },
  balanced: { resolution: 512, steps: 24, cfg: 3.0 },
  quality: { resolution: 640, steps: 32, cfg: 3.5 },
};

const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const layersInput = document.getElementById("layers");
const layersValue = document.getElementById("layers-value");
const presetInput = document.getElementById("inference-preset");
const toggleAdvanced = document.getElementById("toggle-advanced");
const resolutionInput = document.getElementById("resolution");
const stepsInput = document.getElementById("inference-steps");
const cfgInput = document.getElementById("cfg-scale");
const forceCpuInput = document.getElementById("force-cpu");
const decomposeBtn = document.getElementById("decompose-btn");
const resetBtn = document.getElementById("reset-btn");
const progressWrap = document.getElementById("progress-wrap");
const progressFill = document.getElementById("progress-fill");
const progressPercent = document.getElementById("progress-percent");
const progressLabel = document.getElementById("progress-label");
const statusText = document.getElementById("status");
const errorText = document.getElementById("error");
const previewShell = document.getElementById("preview-shell");
const previewImage = document.getElementById("preview-image");
const previewPlaceholder = document.getElementById("preview-placeholder");
const resultsSection = document.getElementById("results");
const gallery = document.getElementById("gallery");
const downloadZip = document.getElementById("download-zip");
const runtimeConsole = document.getElementById("runtime-console");
const clearConsoleBtn = document.getElementById("clear-console");

let selectedFile = null;
let previewUrl = null;
let activeTaskId = null;
let isProcessing = false;
let seenEvents = new Set();

window.addEventListener("error", (event) => {
  appendConsole(`JS ERROR: ${event.message} @ ${event.filename || "unknown"}:${event.lineno || 0}`);
  setStatus("Frontend error detected. Check Runtime Console.");
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event && event.reason ? String(event.reason) : "Unknown promise rejection";
  appendConsole(`UNHANDLED PROMISE: ${reason}`);
  setStatus("Frontend promise error detected. Check Runtime Console.");
});

initialize();

function initialize() {
  if (runtimeConsole) {
    runtimeConsole.textContent = "";
  }
  appendConsole(`Frontend script loaded (build ${APP_BUILD}).`);

  const required = [
    ["fileInput", fileInput],
    ["dropZone", dropZone],
    ["layersInput", layersInput],
    ["layersValue", layersValue],
    ["presetInput", presetInput],
    ["toggleAdvanced", toggleAdvanced],
    ["resolutionInput", resolutionInput],
    ["stepsInput", stepsInput],
    ["cfgInput", cfgInput],
    ["forceCpuInput", forceCpuInput],
    ["decomposeBtn", decomposeBtn],
    ["resetBtn", resetBtn],
    ["progressWrap", progressWrap],
    ["progressFill", progressFill],
    ["progressPercent", progressPercent],
    ["progressLabel", progressLabel],
    ["statusText", statusText],
    ["errorText", errorText],
    ["previewShell", previewShell],
    ["previewImage", previewImage],
    ["previewPlaceholder", previewPlaceholder],
    ["resultsSection", resultsSection],
    ["gallery", gallery],
    ["downloadZip", downloadZip],
  ];
  const missing = required.filter(([, el]) => !el).map(([name]) => name);
  if (missing.length > 0) {
    appendConsole(`Missing DOM nodes: ${missing.join(", ")}`);
    setStatus("UI assets mismatch. Hard refresh (Ctrl+F5) and retry.");
    return;
  }

  layersValue.textContent = layersInput.value;
  presetInput.value = "safe";
  applyPresetValues("safe");
  bindUploadEvents();
  bindActionEvents();
  appendConsole("Console ready.");
}

function bindUploadEvents() {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    handleSelectedFile(file);
  });

  dropZone.addEventListener("click", () => {
    if (!isProcessing) {
      fileInput.click();
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (!isProcessing) {
        dropZone.classList.add("dragover");
      }
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (event) => {
    if (isProcessing) {
      return;
    }
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    handleSelectedFile(file);
  });
}

function bindActionEvents() {
  layersInput.addEventListener("input", () => {
    layersValue.textContent = layersInput.value;
  });

  presetInput.addEventListener("change", () => {
    const preset = presetInput.value;
    toggleAdvanced.checked = preset === "custom";
    setAdvancedEnabled(toggleAdvanced.checked);
    if (preset !== "custom") {
      applyPresetValues(preset);
    }
    appendConsole(`Preset selected: ${preset}.`);
  });

  toggleAdvanced.addEventListener("change", () => {
    setAdvancedEnabled(toggleAdvanced.checked);
    presetInput.value = toggleAdvanced.checked ? "custom" : "balanced";
    if (!toggleAdvanced.checked) {
      applyPresetValues("balanced");
    }
    appendConsole(`Custom tuning ${toggleAdvanced.checked ? "enabled" : "disabled"}.`);
  });

  if (clearConsoleBtn) {
    clearConsoleBtn.addEventListener("click", () => {
      if (runtimeConsole) {
        runtimeConsole.textContent = "";
      }
      appendConsole("Console cleared.");
    });
  }

  resetBtn.addEventListener("click", () => resetState());

  decomposeBtn.addEventListener("click", async () => {
    if (!selectedFile || isProcessing) {
      return;
    }
    await startDecomposition();
  });
}

function setAdvancedEnabled(enabled) {
  resolutionInput.disabled = !enabled;
  stepsInput.disabled = !enabled;
  cfgInput.disabled = !enabled;
}

function applyPresetValues(presetName) {
  const preset = PRESETS[presetName] || PRESETS.balanced;
  resolutionInput.value = String(preset.resolution);
  stepsInput.value = String(preset.steps);
  cfgInput.value = String(preset.cfg);
}

function handleSelectedFile(file) {
  clearError();
  if (!file) {
    selectedFile = null;
    decomposeBtn.disabled = true;
    setStatus("No image selected.");
    return;
  }

  if (!ACCEPTED_MIME_TYPES.has(file.type)) {
    setError("Unsupported format. Upload JPG, PNG, or WEBP.");
    return;
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    setError("File is larger than 20MB.");
    return;
  }

  selectedFile = file;
  decomposeBtn.disabled = false;
  setStatus(`Selected: ${file.name}`);
  appendConsole(`File selected: ${file.name} (${Math.round(file.size / 1024)} KB).`);
  renderPreview(file);
  clearResults();
}

function renderPreview(file) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  previewShell.classList.remove("empty");
}

function collectInferenceOptions() {
  const preset = presetInput.value;
  const deviceMode = forceCpuInput.checked ? "cpu" : "auto";
  const options = {
    inference_preset: preset,
    device_mode: deviceMode,
    resolution: resolutionInput.value,
    num_inference_steps: stepsInput.value,
    true_cfg_scale: cfgInput.value,
    use_en_prompt: "true",
    cfg_normalize: "true",
  };
  return options;
}

async function startDecomposition() {
  if (!selectedFile) {
    return;
  }

  isProcessing = true;
  activeTaskId = null;
  seenEvents = new Set();
  decomposeBtn.disabled = true;
  clearError();
  clearResults();
  showProgress(5, "Uploading image...");
  appendConsole("Upload started.");

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("num_layers", String(layersInput.value));

  const options = collectInferenceOptions();
  Object.entries(options).forEach(([key, value]) => formData.append(key, String(value)));
  appendConsole(
    `Inference config: preset=${options.inference_preset}, mode=${options.device_mode}, resolution=${options.resolution}, steps=${options.num_inference_steps}, cfg=${options.true_cfg_scale}.`
  );

  try {
    const response = await fetch("/api/decompose", {
      method: "POST",
      headers: { "X-Async-Only": "true" },
      body: formData,
    });

    if (!response.ok) {
      const payload = await safeJson(response);
      throw new Error(payload.detail || "Image decomposition failed");
    }

    const payload = await safeJson(response);
    if (!payload.task_id) {
      throw new Error("Backend did not return a task id");
    }

    activeTaskId = payload.task_id;
    appendConsole(`Task created: ${activeTaskId}`);
    await pollTask(activeTaskId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error";
    setError(message);
    appendConsole(`ERROR: ${message}`);
    hideProgress();
  } finally {
    isProcessing = false;
    decomposeBtn.disabled = !selectedFile;
  }
}

async function pollTask(taskId) {
  const startedAt = Date.now();
  let consecutiveStatusFailures = 0;
  let lastSuccessfulStatusAt = Date.now();
  let lastStatus = "";
  let lastProgress = -1;
  let lastMessage = "";
  let lastState = "";
  let lastEventFingerprint = "";
  let lastActivityAt = Date.now();
  let lastHeartbeatLogAt = Date.now();
  let lastStallNoticeAt = 0;

  while (Date.now() - startedAt < PROCESSING_TIMEOUT_MS) {
    let payload;
    try {
      const response = await fetch(`/api/status/${taskId}`, {
        method: "GET",
        cache: "no-store",
      });
      if (!response.ok) {
        const failurePayload = await safeJson(response);
        throw new Error(failurePayload.detail || `Status request failed (${response.status})`);
      }

      payload = await safeJson(response);
      lastSuccessfulStatusAt = Date.now();
      consecutiveStatusFailures = 0;
    } catch (error) {
      consecutiveStatusFailures += 1;
      const message = error instanceof Error ? error.message : "Unknown status error";
      appendConsole(`Status fetch issue (${consecutiveStatusFailures}): ${message}`);
      setStatus("Status connection interrupted. Retrying...");
      if (Date.now() - lastSuccessfulStatusAt > STATUS_FETCH_MAX_DOWNTIME_MS) {
        throw new Error(
          `Status endpoint unreachable for over ${Math.floor(
            STATUS_FETCH_MAX_DOWNTIME_MS / 60000
          )} minutes. Check backend container health. Last error: ${message}`
        );
      }
      await sleep(POLL_INTERVAL_MS);
      continue;
    }

    const progress = typeof payload.progress === "number" ? payload.progress : 0;
    const message = payload.message || `Task ${taskId} is ${payload.status}`;
    const state = payload.state || "";
    const events = Array.isArray(payload.events) ? payload.events : [];
    const eventFingerprint = events.join("|");
    showProgress(progress, message);
    appendEvents(events);

    const changed =
      payload.status !== lastStatus ||
      progress !== lastProgress ||
      message !== lastMessage ||
      state !== lastState ||
      eventFingerprint !== lastEventFingerprint;

    if (changed) {
      lastActivityAt = Date.now();
      lastHeartbeatLogAt = Date.now();
      appendConsole(`Status=${payload.status} state=${state || "n/a"} progress=${progress}% ${message}`);
      lastStatus = payload.status;
      lastProgress = progress;
      lastMessage = message;
      lastState = state;
      lastEventFingerprint = eventFingerprint;
    } else {
      const now = Date.now();
      if (now - lastActivityAt > STALL_NO_ACTIVITY_MS && now - lastStallNoticeAt > NO_EVENT_HEARTBEAT_MS) {
        const idleSec = Math.max(1, Math.floor((now - lastActivityAt) / 1000));
        appendConsole(
          `No new status/event changes for ${idleSec}s, but task is still active. Waiting for next worker heartbeat...`
        );
        setStatus("Processing... first run can take longer while model files are loaded.");
        lastStallNoticeAt = now;
      } else if (now - lastHeartbeatLogAt > NO_EVENT_HEARTBEAT_MS) {
        const elapsedSec = Math.max(1, Math.floor((now - startedAt) / 1000));
        appendConsole(`Still processing... (${elapsedSec}s elapsed, no new worker events yet).`);
        lastHeartbeatLogAt = now;
      }
    }

    if (payload.status === "done") {
      showProgress(100, "Decomposition complete.");
      renderResults(payload);
      setStatus("Layers are ready. Preview and download below.");
      appendConsole("Task completed successfully.");
      return;
    }

    if (payload.status === "error") {
      throw new Error(payload.error || "Task failed");
    }

    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error(
    "Task timed out without completion. Try Safe preset + Force CPU Mode, then check worker logs."
  );
}

function appendEvents(events) {
  if (!Array.isArray(events)) {
    return;
  }
  events.forEach((eventMessage) => {
    const line = String(eventMessage || "").trim();
    if (!line || seenEvents.has(line)) {
      return;
    }
    seenEvents.add(line);
    appendConsole(`Worker: ${line}`);
  });
}

function renderResults(payload) {
  const layerUrls = Array.isArray(payload.layer_urls) ? payload.layer_urls : [];
  if (!layerUrls.length) {
    throw new Error("No layer files were returned by the server.");
  }

  resultsSection.classList.remove("hidden");
  gallery.innerHTML = "";

  layerUrls.forEach((url, index) => {
    const card = document.createElement("article");
    card.className = "layer-card";
    card.style.animationDelay = `${index * 55}ms`;

    const imageWrap = document.createElement("div");
    imageWrap.className = "layer-preview";

    const image = document.createElement("img");
    image.src = url;
    image.alt = `Layer ${index + 1}`;
    image.loading = "lazy";
    imageWrap.appendChild(image);

    const meta = document.createElement("div");
    meta.className = "layer-meta";

    const name = document.createElement("p");
    name.className = "layer-name";
    name.textContent = buildLayerName(index, layerUrls.length);

    const download = document.createElement("a");
    download.className = "btn btn-secondary";
    download.href = url;
    download.download = `layer_${String(index + 1).padStart(2, "0")}.png`;
    download.textContent = "Download Layer";

    meta.append(name, download);
    card.append(imageWrap, meta);
    gallery.appendChild(card);
  });

  if (payload.download_url) {
    downloadZip.href = payload.download_url;
    downloadZip.download = `${activeTaskId || "image"}_layers.zip`;
  } else {
    downloadZip.removeAttribute("href");
  }
}

function buildLayerName(index, total) {
  if (index === 0) {
    return "Layer 1 — Background";
  }
  if (index === total - 1) {
    return `Layer ${index + 1} — Foreground`;
  }
  return `Layer ${index + 1} — Detail`;
}

function showProgress(percent, label) {
  const bounded = Math.max(0, Math.min(100, Math.round(percent)));
  progressWrap.classList.remove("hidden");
  progressFill.style.width = `${bounded}%`;
  progressPercent.textContent = `${bounded}%`;
  progressLabel.textContent = label || "Processing...";
}

function hideProgress() {
  progressWrap.classList.add("hidden");
  progressFill.style.width = "0%";
  progressPercent.textContent = "0%";
  progressLabel.textContent = "Processing...";
}

function setStatus(message) {
  statusText.textContent = message;
}

function setError(message) {
  errorText.textContent = `Error: ${message}`;
  errorText.classList.remove("hidden");
}

function clearError() {
  errorText.textContent = "";
  errorText.classList.add("hidden");
}

function clearResults() {
  resultsSection.classList.add("hidden");
  gallery.innerHTML = "";
  downloadZip.removeAttribute("href");
}

function resetState() {
  selectedFile = null;
  activeTaskId = null;
  isProcessing = false;
  seenEvents = new Set();

  fileInput.value = "";
  decomposeBtn.disabled = true;
  layersInput.value = "4";
  layersValue.textContent = "4";
  presetInput.value = "safe";
  toggleAdvanced.checked = false;
  forceCpuInput.checked = false;
  setAdvancedEnabled(false);
  applyPresetValues("safe");

  clearError();
  setStatus("Ready for a new image.");
  hideProgress();
  clearResults();
  runtimeConsole.textContent = "";
  appendConsole("Reset complete.");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.src = "";
  previewImage.hidden = true;
  previewPlaceholder.hidden = false;
  previewShell.classList.add("empty");
}

function appendConsole(message) {
  const line = String(message || "").trim();
  if (!line) {
    return;
  }
  const timestamp = new Date().toLocaleTimeString();
  if (runtimeConsole) {
    runtimeConsole.textContent += `[${timestamp}] ${line}\n`;
    runtimeConsole.scrollTop = runtimeConsole.scrollHeight;
    return;
  }
  // Fallback so diagnostics are never silently dropped.
  console.log(`[${timestamp}] ${line}`);
}

async function safeJson(response) {
  try {
    return await response.json();
  } catch (_error) {
    try {
      const text = await response.text();
      return { detail: text || "Unexpected non-JSON server response" };
    } catch (_innerError) {
      return {};
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

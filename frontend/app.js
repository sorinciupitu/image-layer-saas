const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;
const ACCEPTED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);
const PROCESSING_TIMEOUT_MS = 30 * 60 * 1000;

const fileInput = document.getElementById("file-input");
const dropZone = document.getElementById("drop-zone");
const layersInput = document.getElementById("layers");
const layersValue = document.getElementById("layers-value");
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

let selectedFile = null;
let previewUrl = null;
let activeTaskId = null;
let isProcessing = false;

initialize();

function initialize() {
  layersValue.textContent = layersInput.value;
  bindUploadEvents();
  bindActionEvents();
}

function bindUploadEvents() {
  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0] ? fileInput.files[0] : null;
    handleSelectedFile(file);
  });

  dropZone.addEventListener("click", () => {
    if (isProcessing) {
      return;
    }
    fileInput.click();
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

  resetBtn.addEventListener("click", () => resetState());

  decomposeBtn.addEventListener("click", async () => {
    if (!selectedFile || isProcessing) {
      return;
    }
    await startDecomposition();
  });
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
  renderPreview(file);
  clearResults();
}

function renderPreview(file) {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
  previewPlaceholder.hidden = true;
  previewShell.classList.remove("empty");
}

async function startDecomposition() {
  if (!selectedFile) {
    return;
  }

  isProcessing = true;
  activeTaskId = null;
  decomposeBtn.disabled = true;
  clearError();
  clearResults();
  showProgress(5, "Uploading image...");

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("num_layers", String(layersInput.value));

  try {
    const response = await fetch("/api/decompose", {
      method: "POST",
      headers: {
        "X-Async-Only": "true",
      },
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
    await pollTask(activeTaskId);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unexpected error";
    setError(message);
    hideProgress();
  } finally {
    isProcessing = false;
    decomposeBtn.disabled = !selectedFile;
  }
}

async function pollTask(taskId) {
  const startedAt = Date.now();
  const timeoutMs = PROCESSING_TIMEOUT_MS;

  while (Date.now() - startedAt < timeoutMs) {
    const response = await fetch(`/api/status/${taskId}`, { method: "GET" });
    if (!response.ok) {
      const payload = await safeJson(response);
      throw new Error(payload.detail || "Failed to fetch task status");
    }

    const payload = await safeJson(response);
    const progress = typeof payload.progress === "number" ? payload.progress : 0;
    const message = payload.message || `Task ${taskId} is ${payload.status}`;
    showProgress(progress, message);

    if (payload.status === "done") {
      showProgress(100, "Decomposition complete.");
      renderResults(payload);
      setStatus("Layers are ready. Preview and download below.");
      return;
    }

    if (payload.status === "error") {
      throw new Error(payload.error || "Task failed");
    }

    await sleep(1500);
  }

  throw new Error(
    "Processing is still running (first model download can take several minutes). Please retry in a moment."
  );
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

  fileInput.value = "";
  decomposeBtn.disabled = true;
  layersInput.value = "4";
  layersValue.textContent = "4";

  clearError();
  setStatus("Ready for a new image.");
  hideProgress();
  clearResults();

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  previewImage.src = "";
  previewImage.hidden = true;
  previewPlaceholder.hidden = false;
  previewShell.classList.add("empty");
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

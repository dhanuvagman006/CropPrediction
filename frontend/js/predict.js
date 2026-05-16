function getApiBase() {
  const params = new URLSearchParams(window.location.search);
  const paramBase = params.get("api");
  if (paramBase) return paramBase.replace(/\/$/, "");
  if (window.location.protocol !== "file:") {
    return `${window.location.origin}/api`;
  }
  return "http://localhost:5000/api";
}

const API_BASE = getApiBase();
let currentCrop = "rice";
let currentModel = "LSTM";
let featureConfig = null;

document.addEventListener("DOMContentLoaded", () => {
  loadFeatureConfig(currentCrop);

  document.querySelectorAll('input[name="crop"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      currentCrop = e.target.value;
      loadFeatureConfig(currentCrop);
    });
  });

  document.querySelectorAll('input[name="model"]').forEach((radio) => {
    radio.addEventListener("change", (e) => {
      currentModel = e.target.value;
    });
  });

  document.getElementById("predict-btn").addEventListener("click", handlePredict);
  document.getElementById("reset-btn").addEventListener("click", resetForm);

  document.querySelectorAll(".param-group-header").forEach((header) => {
    header.addEventListener("click", () => {
      header.parentElement.classList.toggle("collapsed");
    });
  });
});

async function loadFeatureConfig(crop) {
  const container = document.getElementById("params-container");
  container.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 20px;">Loading parameters...</p>';

  try {
    const resp = await fetch(`${API_BASE}/feature-config/${crop}`);
    if (!resp.ok) throw new Error("Failed to load feature config");
    featureConfig = await resp.json();
    renderForm(featureConfig);
  } catch (err) {
    container.innerHTML = `<p style="color: var(--error); text-align: center; padding: 20px;">Error loading parameters: ${err.message}</p>`;
  }
}

function renderForm(config) {
  const container = document.getElementById("params-container");
  const groups = {};

  for (const [key, feat] of Object.entries(config.features)) {
    const group = feat.group || "other";
    if (!groups[group]) groups[group] = [];
    groups[group].push({ key, ...feat });
  }

  const groupLabels = {
    soil: "🌱 Soil Parameters",
    climate: "🌧️ Climate Parameters",
    irrigation: "💧 Irrigation & Farming",
    crop: "🌿 Crop Specific",
  };

  let html = "";
  for (const [groupKey, fields] of Object.entries(groups)) {
    const label = groupLabels[groupKey] || groupKey;
    html += `<div class="param-group" data-group="${groupKey}">`;
    html += `<button class="param-group-header" type="button">${label} <span class="toggle-icon">▼</span></button>`;
    html += `<div class="param-group-body">`;

    for (const field of fields) {
      html += `<div class="form-group">`;
      html += `<label for="feat-${field.key}">${field.label} <span class="tooltip-icon" data-tooltip="${field.tooltip}">?</span></label>`;

      if (field.type === "categorical") {
        html += `<select id="feat-${field.key}" name="${field.key}">`;
        for (const opt of field.options) {
          const selected = opt === field.default ? " selected" : "";
          html += `<option value="${opt}"${selected}>${opt}</option>`;
        }
        html += `</select>`;
      } else {
        html += `<input type="number" id="feat-${field.key}" name="${field.key}" `;
        html += `min="${field.min}" max="${field.max}" step="${field.step}" value="${field.default}" `;
        html += `placeholder="${field.default}" aria-label="${field.label}">`;
        html += `<span class="error-msg">Please enter a valid value</span>`;
      }

      html += `</div>`;
    }

    html += `</div></div>`;
  }

  container.innerHTML = html;
}

async function handlePredict() {
  const btn = document.getElementById("predict-btn");
  const loading = document.getElementById("loading");
  const errorAlert = document.getElementById("error-alert");
  const resultContent = document.getElementById("result-content");
  const placeholder = document.getElementById("result-placeholder");

  errorAlert.classList.remove("show");
  document.querySelectorAll(".form-group input").forEach((inp) => inp.classList.remove("error"));

  let hasError = false;
  document.querySelectorAll('.form-group input[type="number"]').forEach((inp) => {
    const val = inp.value.trim();
    if (val === "" || isNaN(val)) {
      inp.classList.add("error");
      hasError = true;
    }
  });

  if (hasError) {
    errorAlert.textContent = "Please fill in all numeric fields with valid numbers.";
    errorAlert.classList.add("show");
    return;
  }

  const features = {};
  for (const [key, feat] of Object.entries(featureConfig.features)) {
    const el = document.getElementById(`feat-${key}`);
    if (!el) continue;

    if (feat.type === "categorical") {
      let val = el.value;
      if (key === "farm_mechanization") {
        val = val.includes("1") ? 1 : 0;
      } else if (key === "crop_variety") {
        val = val.includes("HYV") || val === "1" ? 1 : 0;
      }
      features[key] = val;
    } else {
      features[key] = parseFloat(el.value);
    }
  }

  btn.disabled = true;
  btn.textContent = "Predicting...";
  loading.classList.add("show");
  resultContent.classList.remove("show");
  placeholder.style.display = "block";

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crop: currentCrop,
        model: currentModel,
        features: features,
      }),
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.error || "Prediction request failed");
    }

    const result = await resp.json();
    displayResult(result);
  } catch (err) {
    errorAlert.textContent = `Prediction failed: ${err.message}`;
    errorAlert.classList.add("show");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict Yield";
    loading.classList.remove("show");
  }
}

function displayResult(result) {
  const placeholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");

  placeholder.style.display = "none";

  document.getElementById("result-value").textContent = formatNumber(result.predicted_yield);
  document.getElementById("result-crop").textContent = capitalize(result.crop);
  document.getElementById("result-model").textContent = result.model;
  document.getElementById("ci-lower").textContent = formatNumber(result.confidence_interval.lower);
  document.getElementById("ci-upper").textContent = formatNumber(result.confidence_interval.upper);

  const catEl = document.getElementById("result-category");
  catEl.textContent = result.yield_category;
  catEl.className = "category-badge category-" + result.yield_category.toLowerCase();

  const interp = `Based on the entered parameters, the predicted yield for ${capitalize(result.crop)} using ${result.model} is ${formatNumber(result.predicted_yield)} kg/ha. This falls in the <strong>${result.yield_category}</strong> category for the Dakshina Kannada region.`;
  document.getElementById("result-interpretation").innerHTML = interp;

  resultContent.classList.add("show");
}

function resetForm() {
  document.getElementById("result-content").classList.remove("show");
  document.getElementById("result-placeholder").style.display = "block";
  document.getElementById("error-alert").classList.remove("show");

  document.querySelectorAll('.form-group input[type="number"]').forEach((inp) => {
    inp.classList.remove("error");
  });

  window.scrollTo({ top: 0, behavior: "smooth" });
}

function formatNumber(num) {
  return Number(num).toLocaleString("en-IN", { maximumFractionDigits: 1 });
}

function capitalize(str) {
  return str.split(" ").map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

const defaultCrop = "rice";
const defaultModel = "LSTM";

function toTitleCase(value) {
  return value.replace(/\b\w/g, (char) => char.toUpperCase());
}

function setDefaultsFromQuery() {
  const params = new URLSearchParams(window.location.search);
  const crop = params.get("crop") || defaultCrop;
  const model = params.get("model") || defaultModel;

  document.getElementById("crop-select").value = crop;
  document.getElementById("model-select").value = model;
}

async function loadPlots() {
  const crop = document.getElementById("crop-select").value;
  const model = document.getElementById("model-select").value;
  const basePath = `assets/evaluation/${crop}_${model}`;

  const summary = document.getElementById("evaluation-summary");
  const grid = document.getElementById("evaluation-grid");
  const hint = document.getElementById("evaluation-hint");

  summary.innerHTML = "";
  grid.innerHTML = "<div class=\"evaluation-loading\">Loading plots...</div>";
  hint.textContent = `Looking for ${crop} / ${model} in ${basePath}.`;

  try {
    const response = await fetch(`${basePath}/plots.json`);
    if (!response.ok) {
      throw new Error("Plots manifest not found. Run the evaluation script first.");
    }
    const data = await response.json();

    summary.innerHTML = renderSummary(data);
    grid.innerHTML = renderPlots(data, basePath);
  } catch (error) {
    summary.innerHTML = "";
    grid.innerHTML = `<div class=\"evaluation-empty\">${error.message}</div>`;
  }
}

function renderSummary(data) {
  const metrics = data.metrics || {};
  const notes = data.notes || [];
  const items = [
    { label: "R2", value: metrics.r2 },
    { label: "RMSE", value: metrics.rmse },
    { label: "MAE", value: metrics.mae },
    { label: "MAPE", value: metrics.mape, suffix: "%" },
    { label: "Accuracy", value: metrics.accuracy_percent, suffix: "%" },
  ];

  const metricHtml = items
    .filter((item) => item.value !== undefined)
    .map((item) => {
      const suffix = item.suffix || "";
      return `<div class=\"metric-chip\"><span>${item.label}</span><strong>${item.value}${suffix}</strong></div>`;
    })
    .join("");

  const noteHtml = notes.length
    ? `<div class=\"evaluation-notes\">${notes.map((n) => `<span>${n}</span>`).join("")}</div>`
    : "";

  return `
    <div class=\"evaluation-summary-card\">
      <div>
        <h3>${toTitleCase(data.crop)} — ${data.model}</h3>
        <p>Test-set evaluation metrics</p>
      </div>
      <div class=\"metric-row\">${metricHtml}</div>
      ${noteHtml}
    </div>
  `;
}

function renderPlots(data, basePath) {
  if (!data.plots || !data.plots.length) {
    return `<div class=\"evaluation-empty\">No plots found in manifest.</div>`;
  }

  return data.plots
    .map((plot) => {
      const imgSrc = `${basePath}/${plot.file}`;
      const description = plot.description || "";
      return `
        <div class=\"plot-card\">
          <div class=\"plot-image\">
            <img src=\"${imgSrc}\" alt=\"${plot.title}\">
          </div>
          <div class=\"plot-meta\">
            <h4>${plot.title}</h4>
            <p>${description}</p>
          </div>
        </div>
      `;
    })
    .join("");
}

setDefaultsFromQuery();

document.getElementById("load-plots").addEventListener("click", () => {
  loadPlots();
});

window.addEventListener("load", () => {
  loadPlots();
});

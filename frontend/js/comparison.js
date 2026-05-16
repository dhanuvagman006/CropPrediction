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
let comparisonData = null;
let allMetrics = null;
let selectedCrop = "all";
let sortColumn = "avg_r2";
let sortDirection = "desc";
let charts = {};

const CROP_NAMES = {
  "rice": "Rice",
  "coconut": "Coconut",
  "arecanut": "Arecanut",
  "banana": "Banana",
  "black pepper": "Black Pepper",
  "cashew": "Cashew",
  "cocoa": "Cocoa",
  "sweet potato": "Sweet Potato",
};

const MODEL_COLORS = {
  LSTM: "#1B4332",
  BiLSTM: "#2D6A4F",
  GRU: "#40916C",
  "CNN-LSTM": "#52B788",
  Transformer: "#D4A017",
  Autoencoder: "#B8860B",
};

document.addEventListener("DOMContentLoaded", () => {
  loadData();

  document.querySelectorAll(".crop-tab").forEach((tab) => {
    tab.addEventListener("click", (e) => {
      document.querySelectorAll(".crop-tab").forEach((t) => t.classList.remove("active"));
      e.target.classList.add("active");
      selectedCrop = e.target.dataset.crop;
      updatePage();
    });
  });

  document.querySelectorAll("#metrics-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (sortColumn === col) {
        sortDirection = sortDirection === "desc" ? "asc" : "desc";
      } else {
        sortColumn = col;
        sortDirection = "desc";
      }
      document.querySelectorAll(".sort-arrow").forEach((s) => {
        s.textContent = "";
        s.classList.remove("active");
      });
      const arrow = th.querySelector(".sort-arrow");
      if (arrow) {
        arrow.textContent = sortDirection === "desc" ? "▼" : "▲";
        arrow.classList.add("active");
      }
      renderMetricsTable();
    });
  });
});

async function loadData() {
  try {
    const [metricsResp, compResp] = await Promise.all([
      fetch(`${API_BASE}/metrics`),
      fetch(`${API_BASE}/comparison-data`),
    ]);

    if (!metricsResp.ok || !compResp.ok) {
      throw new Error("Failed to load data. Ensure the backend is running and models are trained.");
    }

    allMetrics = await metricsResp.json();
    comparisonData = await compResp.json();

    updatePage();
  } catch (err) {
    document.getElementById("summary-cards").innerHTML =
      `<div style="text-align:center;padding:40px;grid-column:1/-1;color:var(--error);">${err.message}</div>`;
  }
}

function updatePage() {
  renderSummaryCards();
  renderMetricsTable();
  renderBarChart();
  renderRadarChart();
  renderLineChart();
  renderScatterChart();
  renderHeatmap();
  updateRecommendation();
}

function getFilteredMetrics() {
  if (!allMetrics) return {};
  if (selectedCrop === "all") return allMetrics;
  const filtered = {};
  if (allMetrics[selectedCrop]) {
    filtered[selectedCrop] = allMetrics[selectedCrop];
  }
  return filtered;
}

function getModelAverages(filteredMetrics) {
  const models = comparisonData ? comparisonData.models : ["LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer", "Autoencoder"];
  const crops = comparisonData ? comparisonData.crops : Object.keys(filteredMetrics);

  const avgs = {};
  for (const model of models) {
    let totalR2 = 0, totalAcc = 0, count = 0;
    for (const crop of crops) {
      if (filteredMetrics[crop] && filteredMetrics[crop][model]) {
        totalR2 += filteredMetrics[crop][model].r2;
        totalAcc += filteredMetrics[crop][model].accuracy_percent;
        count++;
      }
    }
    avgs[model] = {
      r2: count > 0 ? (totalR2 / count) : 0,
      accuracy: count > 0 ? (totalAcc / count) : 0,
    };
  }
  return avgs;
}

function renderSummaryCards() {
  const container = document.getElementById("summary-cards");
  const filtered = getFilteredMetrics();
  const avgs = getModelAverages(filtered);
  const models = Object.keys(avgs);

  const bestModel = models.reduce((a, b) => (avgs[a].r2 > avgs[b].r2 ? a : b), models[0]);

  let html = "";
  for (const model of models) {
    const isBest = model === bestModel;
    html += `<div class="summary-card${isBest ? " best" : ""}">`;
    if (isBest) html += `<div class="best-badge">Best Model</div>`;
    html += `<h4>${model}</h4>`;
    html += `<div class="summary-stat"><div class="stat-val">${avgs[model].r2.toFixed(3)}</div><div class="stat-lbl">Avg R² Score</div></div>`;
    html += `<div class="summary-stat"><div class="stat-val">${avgs[model].accuracy.toFixed(1)}%</div><div class="stat-lbl">Avg Accuracy</div></div>`;
    html += `</div>`;
  }

  container.innerHTML = html;
}

function renderMetricsTable() {
  const tbody = document.getElementById("metrics-tbody");
  if (!comparisonData || !comparisonData.metrics_table) return;

  let rows = [...comparisonData.metrics_table];
  rows.sort((a, b) => {
    const aVal = a[sortColumn] || 0;
    const bVal = b[sortColumn] || 0;
    return sortDirection === "desc" ? bVal - aVal : aVal - bVal;
  });

  let html = "";
  rows.forEach((row, idx) => {
    const r2 = row.avg_r2 || 0;
    let r2Class = "";
    if (r2 >= 0.90) r2Class = "color: var(--emerald); font-weight: 700;";
    else if (r2 >= 0.85) r2Class = "color: var(--success); font-weight: 600;";
    else r2Class = "color: var(--warning); font-weight: 600;";

    html += `<tr>`;
    html += `<td><strong>${row.model}</strong></td>`;
    html += `<td style="${r2Class}">${(row.avg_r2 || 0).toFixed(4)}</td>`;
    html += `<td>${(row.avg_rmse || 0).toFixed(2)}</td>`;
    html += `<td>${(row.avg_mae || 0).toFixed(2)}</td>`;
    html += `<td>${(row.avg_mape || 0).toFixed(2)}</td>`;
    html += `<td>${(row.avg_nse || 0).toFixed(4)}</td>`;
    html += `<td>${(row.avg_accuracy || 0).toFixed(2)}%</td>`;
    html += `<td>${(row.avg_size || 0).toFixed(1)}</td>`;
    html += `<td>#${idx + 1}</td>`;
    html += `</tr>`;
  });

  tbody.innerHTML = html;
}

function renderBarChart() {
  const ctx = document.getElementById("bar-chart");
  if (!ctx) return;
  if (charts.bar) charts.bar.destroy();

  const models = comparisonData.models;
  const crops = comparisonData.crops;
  const datasets = [];

  const cropColors = {
    rice: "#1B4332", coconut: "#2D6A4F", arecanut: "#40916C", banana: "#52B788",
    "black pepper": "#D4A017", cashew: "#B8860B", cocoa: "#8B6914", "sweet potato": "#CD853F",
  };

  const visibleCrops = selectedCrop === "all" ? crops : [selectedCrop];

  for (const crop of visibleCrops) {
    const data = [];
    for (const model of models) {
      const val = allMetrics && allMetrics[crop] && allMetrics[crop][model]
        ? allMetrics[crop][model].accuracy_percent : 0;
      data.push(val);
    }
    datasets.push({
      label: CROP_NAMES[crop] || crop,
      data: data,
      backgroundColor: cropColors[crop] || "#999",
      borderRadius: 4,
    });
  }

  charts.bar = new Chart(ctx, {
    type: "bar",
    data: { labels: models, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%`,
          },
        },
      },
      scales: {
        y: { beginAtZero: false, title: { display: true, text: "Accuracy (%)" } },
      },
    },
  });
}

function renderRadarChart() {
  const ctx = document.getElementById("radar-chart");
  if (!ctx) return;
  if (charts.radar) charts.radar.destroy();

  const models = comparisonData.models;
  const radarData = comparisonData.radar_data;

  const datasets = [];
  for (const model of models) {
    const rd = radarData[model] || {};
    datasets.push({
      label: model,
      data: [
        rd.r2 || 0,
        rd.nse || 0,
        (rd.accuracy || 0) / 100,
        (rd.mape_inv || 0) / 100,
        1 - (rd.rmse_norm || 0),
      ],
      borderColor: MODEL_COLORS[model] || "#999",
      backgroundColor: (MODEL_COLORS[model] || "#999") + "33",
      pointBackgroundColor: MODEL_COLORS[model] || "#999",
      borderWidth: 2,
    });
  }

  charts.radar = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["R²", "NSE", "Accuracy (norm)", "MAPE⁻¹ (norm)", "RMSE⁻¹ (norm)"],
      datasets: datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          ticks: { stepSize: 0.2 },
        },
      },
    },
  });
}

function renderLineChart() {
  const ctx = document.getElementById("line-chart");
  if (!ctx) return;
  if (charts.line) charts.line.destroy();

  const models = comparisonData.models;
  const crops = comparisonData.crops;

  const datasets = [];
  for (const model of models) {
    const data = [];
    for (const crop of crops) {
      const val = allMetrics && allMetrics[crop] && allMetrics[crop][model]
        ? allMetrics[crop][model].r2 : 0;
      data.push(val);
    }
    datasets.push({
      label: model,
      data: data,
      borderColor: MODEL_COLORS[model] || "#999",
      backgroundColor: (MODEL_COLORS[model] || "#999") + "22",
      tension: 0.3,
      pointRadius: 5,
      pointHoverRadius: 7,
      borderWidth: 2,
    });
  }

  const cropLabels = crops.map((c) => CROP_NAMES[c] || c);

  charts.line = new Chart(ctx, {
    type: "line",
    data: { labels: cropLabels, datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: R² = ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        y: { min: 0.5, max: 1, title: { display: true, text: "R² Score" } },
      },
    },
  });
}

function renderScatterChart() {
  const ctx = document.getElementById("scatter-chart");
  if (!ctx) return;
  if (charts.scatter) charts.scatter.destroy();

  const models = comparisonData.models;
  const crops = comparisonData.crops;

  const cropColors = {
    rice: "#1B4332", coconut: "#2D6A4F", arecanut: "#40916C", banana: "#52B788",
    "black pepper": "#D4A017", cashew: "#B8860B", cocoa: "#8B6914", "sweet potato": "#CD853F",
  };

  const datasets = [];
  for (const model of models) {
    const points = [];
    for (const crop of crops) {
      if (allMetrics && allMetrics[crop] && allMetrics[crop][model]) {
        points.push({
          x: allMetrics[crop][model].rmse,
          y: allMetrics[crop][model].r2,
        });
      }
    }
    datasets.push({
      label: model,
      data: points,
      backgroundColor: MODEL_COLORS[model] || "#999",
      borderColor: MODEL_COLORS[model] || "#999",
      pointRadius: 7,
      pointHoverRadius: 9,
    });
  }

  charts.scatter = new Chart(ctx, {
    type: "scatter",
    data: { datasets: datasets },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: (ctx) => `RMSE: ${ctx.parsed.x.toFixed(2)}, R²: ${ctx.parsed.y.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "RMSE (kg/ha)" } },
        y: { title: { display: true, text: "R² Score" }, min: 0.5, max: 1 },
      },
    },
  });
}

function renderHeatmap() {
  const container = document.getElementById("heatmap-container");
  if (!comparisonData) return;

  const models = comparisonData.models;
  const heatmapData = comparisonData.heatmap_data;

  let html = `<table class="heatmap-table">`;
  html += `<thead><tr><th>Crop</th>`;
  for (const model of models) {
    html += `<th>${model}</th>`;
  }
  html += `</tr></thead><tbody>`;

  for (const row of heatmapData) {
    const cropKey = row.crop;
    const cropLabel = CROP_NAMES[cropKey] || cropKey;
    html += `<tr><td>${cropLabel}</td>`;

    for (const model of models) {
      const val = row[model] || 0;
      let cls = "heat-low";
      if (val >= 95) cls = "heat-best";
      else if (val >= 90) cls = "heat-high";
      else if (val >= 85) cls = "heat-mid";

      html += `<td class="${cls}">${val.toFixed(1)}%</td>`;
    }
    html += `</tr>`;
  }

  html += `</tbody></table>`;
  container.innerHTML = html;
}

function updateRecommendation() {
  const el = document.getElementById("recommendation-text");
  if (!allMetrics) {
    el.textContent = "Loading...";
    return;
  }

  const crops = selectedCrop === "all"
    ? comparisonData.crops
    : [selectedCrop];

  let bestModel = null;
  let bestR2 = -1;
  let bestAcc = 0;
  let bestNse = 0;

  const models = comparisonData.models;

  for (const model of models) {
    let totalR2 = 0, totalAcc = 0, totalNse = 0, count = 0;
    for (const crop of crops) {
      if (allMetrics[crop] && allMetrics[crop][model]) {
        totalR2 += allMetrics[crop][model].r2;
        totalAcc += allMetrics[crop][model].accuracy_percent;
        totalNse += allMetrics[crop][model].nse;
        count++;
      }
    }
    if (count > 0) {
      const avgR2 = totalR2 / count;
      if (avgR2 > bestR2) {
        bestR2 = avgR2;
        bestModel = model;
        bestAcc = totalAcc / count;
        bestNse = totalNse / count;
      }
    }
  }

  const cropLabel = selectedCrop === "all"
    ? "all crops"
    : CROP_NAMES[selectedCrop] || selectedCrop;

  if (bestModel) {
    el.innerHTML = `For <strong>${cropLabel}</strong>, the best performing model is <strong>${bestModel}</strong> with ` +
      `R² = <strong>${bestR2.toFixed(4)}</strong>, Accuracy = <strong>${bestAcc.toFixed(2)}%</strong>, ` +
      `and NSE = <strong>${bestNse.toFixed(4)}</strong>. This model is recommended for production use.`;
  } else {
    el.textContent = "Insufficient data to make a recommendation.";
  }
}

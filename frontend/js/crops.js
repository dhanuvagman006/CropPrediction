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
let allCrops = [];

const SEASONAL_DATA = {
  rice: { emoji: "🌾", sowing: [5, 6], growing: [7, 8, 9, 10], harvest: [11] },
  coconut: { emoji: "🥥", sowing: [], growing: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], harvest: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] },
  arecanut: { emoji: "🌴", sowing: [], growing: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], harvest: [2, 3, 8, 9, 10] },
  banana: { emoji: "🍌", sowing: [0, 1, 6, 7], growing: [2, 3, 4, 5, 8, 9], harvest: [4, 5, 10, 11] },
  "black pepper": { emoji: "🌶️", sowing: [5, 6], growing: [7, 8, 9, 10, 11], harvest: [11, 0, 1] },
  cashew: { emoji: "🥜", sowing: [], growing: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], harvest: [1, 2, 3, 4] },
  cocoa: { emoji: "🍫", sowing: [], growing: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], harvest: [4, 5, 9, 10, 11] },
  "sweet potato": { emoji: "🍠", sowing: [9, 10], growing: [10, 11, 0], harvest: [0, 1] },
};

document.addEventListener("DOMContentLoaded", () => {
  loadCrops();

  document.getElementById("search-input").addEventListener("input", (e) => {
    filterCrops(e.target.value.toLowerCase());
  });
});

async function loadCrops() {
  const tbody = document.getElementById("crops-tbody");
  tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:40px;color:var(--text-muted);">Loading crop data...</td></tr>';

  try {
    const resp = await fetch(`${API_BASE}/crops`);
    if (!resp.ok) throw new Error("Failed to load crop data");
    allCrops = await resp.json();
    renderTable(allCrops);
    renderCalendar();
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:40px;color:var(--error);">Error: ${err.message}</td></tr>`;
  }
}

function renderTable(crops) {
  const tbody = document.getElementById("crops-tbody");

  if (crops.length === 0) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:40px;color:var(--text-muted);">No crops match your search.</td></tr>';
    return;
  }

  let html = "";
  for (const crop of crops) {
    const yieldRange = crop.yield_max_kg_ha - crop.yield_min_kg_ha;
    const barWidth = 100;

    html += `<tr>`;
    html += `<td><div class="crop-name-cell"><span class="crop-emoji">${crop.icon}</span>${crop.name}</div></td>`;
    html += `<td>${crop.season}</td>`;
    html += `<td>${crop.soil_type}</td>`;
    html += `<td>${formatNumber(crop.yield_min_kg_ha)} – ${formatNumber(crop.yield_max_kg_ha)}</td>`;
    html += `<td>${crop.optimal_rainfall_mm}</td>`;
    html += `<td>${crop.optimal_temp_C}</td>`;
    html += `<td>${crop.districts}</td>`;
    html += `<td>${crop.description}</td>`;
    html += `</tr>`;
  }

  tbody.innerHTML = html;
}

function filterCrops(query) {
  if (!query) {
    renderTable(allCrops);
    return;
  }
  const filtered = allCrops.filter((c) =>
    c.name.toLowerCase().includes(query) ||
    c.season.toLowerCase().includes(query) ||
    c.soil_type.toLowerCase().includes(query) ||
    c.districts.toLowerCase().includes(query)
  );
  renderTable(filtered);
}

function renderCalendar() {
  const grid = document.getElementById("calendar-grid");
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const cropColors = {
    rice: "#1B4332",
    coconut: "#2D6A4F",
    arecanut: "#40916C",
    banana: "#52B788",
    "black pepper": "#D4A017",
    cashew: "#B8860B",
    cocoa: "#8B4513",
    "sweet potato": "#CD853F",
  };

  for (const crop of allCrops) {
    const cropKey = crop.key;
    const data = SEASONAL_DATA[cropKey];
    if (!data) continue;

    let rowHtml = `<div class="calendar-row">`;
    rowHtml += `<div class="calendar-label">${crop.icon} ${crop.name}</div>`;

    for (let m = 0; m < 12; m++) {
      let cls = "";
      if (data.harvest.includes(m)) {
        cls = "calendar-cell harvest";
      } else if (data.growing.includes(m)) {
        cls = "calendar-cell growing";
      } else if (data.sowing.includes(m)) {
        cls = "calendar-cell sowing";
      }

      if (cls) {
        rowHtml += `<div class="${cls}" style="background: ${cropColors[cropKey] || '#999'}; opacity: ${data.harvest.includes(m) ? 1 : data.growing.includes(m) ? 0.5 : 0.25};" title="${monthNames[m]}"></div>`;
      } else {
        rowHtml += `<div class="calendar-cell"></div>`;
      }
    }

    rowHtml += `</div>`;
    grid.innerHTML += rowHtml;
  }
}

function formatNumber(num) {
  return Number(num).toLocaleString("en-IN");
}

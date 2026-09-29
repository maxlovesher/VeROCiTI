// Camera-network analytics & enforcement (server: city flow model/features_api.py).
const API = import.meta.env.VITE_CITYFLOW_API || '/api';

async function getJson(url) {
  const res = await fetch(url);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) throw new Error(data.error || `Insights API error: ${res.status}`);
  return data;
}

export function fetchInsightsSummary(hours = 24) {
  return getJson(`${API}/insights/summary?hours=${encodeURIComponent(hours)}`);
}

export function lookupPlate(plate) {
  return getJson(`${API}/insights/plate_match?plate=${encodeURIComponent(plate)}`);
}

export async function applyRetentionLimit() {
  const res = await fetch(`${API}/insights/privacy/purge`, { method: 'POST' });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) throw new Error(data.error || `Insights API error: ${res.status}`);
  return data;
}

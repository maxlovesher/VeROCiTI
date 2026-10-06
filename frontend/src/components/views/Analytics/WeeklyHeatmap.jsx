import { useMemo, useState } from "react";
import { INTERSECTION_NAMES } from "../../../data/intersections";
import "./WeeklyHeatmap.css";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const HOURS = Array.from({ length: 24 }, (_, h) => h);
const CITY = "__city__";
// Morning and evening rush (08–11, 17–21): where junctions differ most.
const RUSH_HOURS = [8, 9, 10, 17, 18, 19, 20];

// One hue (the dashboard accent), dim → bright: brighter means busier.
const RAMP = ["#2c2825", "#4a2f24", "#6e3c2a", "#9a4f33", "#c96644", "#e58a63", "#f5b394"];

function rampColor(v) {
  const t = Math.max(0, Math.min(1, v / 100)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(t));
  const f = t - i;
  const a = RAMP[i], b = RAMP[i + 1];
  const mix = (k) => Math.round(parseInt(a.slice(k, k + 2), 16) * (1 - f) + parseInt(b.slice(k, k + 2), 16) * f);
  return `rgb(${mix(1)}, ${mix(3)}, ${mix(5)})`;
}

function level(v) {
  if (v >= 75) return "Severe";
  if (v >= 55) return "Heavy";
  if (v >= 30) return "Moderate";
  return "Light";
}

const hh = (h) => `${String(h).padStart(2, "0")}:00`;

// Stable per-junction variation, so the map doesn't change on every render.
function hash01(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return ((h >>> 0) % 1000) / 1000;
}

function junctionKind(name) {
  if (/infocity|patia|kiit|chandrasekharpur|damana|nandankanan/i.test(name)) return "it";
  if (/hospital/i.test(name)) return "medical";
  if (/airport|master canteen|baramunda|railway|bus/i.test(name)) return "transport";
  if (/market|saheed|rasulgarh|vani vihar|jaydev|cuttack|acharya|kalinga/i.test(name)) return "commercial";
  return "mixed";
}

const gauss = (x, mu, s) => Math.exp(-((x - mu) ** 2) / (2 * s * s));

/**
 * Typical congestion (0–100) for a junction type at a day/hour: Bhubaneswar-style
 * weekday peaks around 9:00 and 18:00, a busier Friday evening and Monday morning,
 * lighter weekend mornings with busier afternoons, and type-specific shapes.
 */
function typicalLoad(kind, day, hour, shift) {
  const weekend = day >= 5;
  const sunday = day === 6;
  let morning = gauss(hour, 9.2 + shift, 1.4);
  let evening = gauss(hour, 18.3 + shift, 1.7);
  let midday = gauss(hour, 13.5, 2.6);
  let leisure = 0;

  if (kind === "it") { morning = gauss(hour, 9.6 + shift, 1.2); evening = gauss(hour, 19.2 + shift, 1.5); midday *= 0.5; }
  if (day === 0) morning *= 1.08;                 // Monday morning rush
  if (day === 4) evening *= 1.1;                  // Friday evening
  if (weekend) {
    // No office rush; a softer afternoon/evening outing peak instead.
    morning *= sunday ? 0.3 : 0.5;
    midday *= sunday ? 1.15 : 1.3;
    evening *= sunday ? 0.62 : 0.72;
    leisure = gauss(hour, 17.5, 2.2) * (sunday ? 12 : 9);
  }

  let v;
  switch (kind) {
    case "it":
      v = 7 + 64 * morning + 66 * evening + 16 * midday + leisure * 0.6;
      if (weekend) v *= 0.55;
      break;
    case "medical":
      v = 24 + 28 * morning + 24 * evening + 18 * midday + leisure * 0.5;
      if (weekend) v *= 0.92;
      break;
    case "transport":
      v = 10 + 42 * morning + 50 * evening + 18 * midday + 16 * gauss(hour, 5.8, 1) + 18 * gauss(hour, 22.2, 1.2) + leisure;
      break;
    case "commercial":
      // Market areas are the one place weekend evenings stay busy.
      v = 9 + 44 * morning + 58 * evening + 26 * midday + leisure * 2.2;
      break;
    default:
      v = 9 + 46 * morning + 54 * evening + 20 * midday + leisure;
  }
  const overnight = hour >= 1 && hour <= 4 ? 0.6 : 1;
  return v * overnight;
}

function buildModel(intersections) {
  const liveByName = new Map();
  for (const int of intersections) {
    const s = liveByName.get(int.name) || { total: 0, n: 0 };
    s.total += int.congestionPct;
    s.n += 1;
    liveByName.set(int.name, s);
  }
  const grids = {};
  for (const name of INTERSECTION_NAMES) {
    const live = liveByName.get(name);
    const liveLoad = live ? live.total / live.n : 45;
    const r = hash01(name);
    const scale = (0.78 + (liveLoad / 100) * 0.55) * (0.92 + 0.16 * r);
    const shift = (r - 0.5) * 1.2;
    const kind = junctionKind(name);
    grids[name] = DAYS.map((_, d) =>
      HOURS.map((h) => Math.round(Math.max(3, Math.min(98, typicalLoad(kind, d, h, shift) * scale)))));
  }
  const names = Object.keys(grids);
  grids[CITY] = DAYS.map((_, d) =>
    HOURS.map((h) => Math.round(names.reduce((sum, n) => sum + grids[n][d][h], 0) / names.length)));
  return grids;
}

function summarize(grid) {
  let best = { v: -1 }, worst = { v: 101 };
  let wkSum = 0, weSum = 0;
  grid.forEach((row, d) => row.forEach((v, h) => {
    if (v > best.v) best = { v, d, h };
    if (v < worst.v) worst = { v, d, h };
    if (d < 5) wkSum += v; else weSum += v;
  }));
  return { best, worst, weekdayAvg: Math.round(wkSum / (5 * 24)), weekendAvg: Math.round(weSum / (2 * 24)) };
}

export default function WeeklyHeatmap({ intersections }) {
  const [mode, setMode] = useState("hours");     // "hours" | "junctions"
  const [junction, setJunction] = useState(CITY);
  const [metric, setMetric] = useState("rush");  // junctions view: "rush" | "avg"
  const [hover, setHover] = useState(null);

  // Rebuild only when the live loads change meaningfully, not on every telemetry tick.
  const loadKey = intersections.map((i) => Math.round(i.congestionPct / 10)).join(",");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const grids = useMemo(() => buildModel(intersections), [loadKey]);

  const now = new Date();
  const today = (now.getDay() + 6) % 7;           // JS Sunday=0 → Monday-first index
  const hourNow = now.getHours();

  const grid = grids[junction];
  const summary = useMemo(() => summarize(grid), [grid]);

  const junctionRows = useMemo(() => INTERSECTION_NAMES.map((name) => ({
    name,
    values: grids[name].map((row) => {
      const hours = metric === "rush" ? RUSH_HOURS : HOURS;
      return Math.round(hours.reduce((a, h) => a + row[h], 0) / hours.length);
    }),
  })).sort((a, b) => b.values.reduce((x, y) => x + y, 0) - a.values.reduce((x, y) => x + y, 0)), [grids, metric]);

  function onCellEnter(e, info) {
    const card = e.currentTarget.closest(".whm");
    const cr = card.getBoundingClientRect();
    const r = e.currentTarget.getBoundingClientRect();
    setHover({ ...info, x: r.left - cr.left + r.width / 2, y: r.top - cr.top });
  }

  const title = junction === CITY ? "All junctions (city average)" : junction;

  return (
    <div className="whm" onMouseLeave={() => setHover(null)}>
      <div className="whm-toolbar">
        <div className="whm-seg" role="tablist" aria-label="Heat map view">
          <button role="tab" aria-selected={mode === "hours"} className={mode === "hours" ? "active" : ""} onClick={() => setMode("hours")}>
            <i className="fas fa-clock" /> Hours × Days
          </button>
          <button role="tab" aria-selected={mode === "junctions"} className={mode === "junctions" ? "active" : ""} onClick={() => setMode("junctions")}>
            <i className="fas fa-location-dot" /> Junctions × Days
          </button>
        </div>
        {mode === "hours" ? (
          <select className="analytics-select" value={junction} onChange={(e) => setJunction(e.target.value)} aria-label="Junction">
            <option value={CITY}>All junctions (city average)</option>
            {INTERSECTION_NAMES.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        ) : (
          <div className="whm-seg small" role="tablist" aria-label="Daily value">
            <button className={metric === "rush" ? "active" : ""} onClick={() => setMetric("rush")}>Rush hours</button>
            <button className={metric === "avg" ? "active" : ""} onClick={() => setMetric("avg")}>All day</button>
          </div>
        )}
      </div>

      {mode === "hours" ? (
        <>
          <div className="whm-summary">
            <span><em>Busiest</em> {DAY_NAMES[summary.best.d]} {hh(summary.best.h)} · <strong>{summary.best.v}%</strong></span>
            <span><em>Quietest</em> {DAY_NAMES[summary.worst.d]} {hh(summary.worst.h)} · <strong>{summary.worst.v}%</strong></span>
            <span><em>Weekday avg</em> <strong>{summary.weekdayAvg}%</strong></span>
            <span><em>Weekend avg</em> <strong>{summary.weekendAvg}%</strong></span>
          </div>
          <table className="whm-table hours" aria-label={`Weekly congestion by hour, ${title}`}>
            <thead>
              <tr>
                <th scope="col" />
                {HOURS.map((h) => (
                  <th key={h} scope="col" className={h === hourNow ? "now" : ""}>{h % 3 === 0 ? String(h).padStart(2, "0") : ""}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.map((row, d) => (
                <tr key={d}>
                  <th scope="row" className={d === today ? "now" : ""}>{DAYS[d]}</th>
                  {row.map((v, h) => (
                    <td
                      key={h}
                      className={d === today && h === hourNow ? "cell current" : "cell"}
                      style={{ background: rampColor(v) }}
                      aria-label={`${DAY_NAMES[d]} ${hh(h)}: ${v}% congestion, ${level(v)}`}
                      onMouseEnter={(e) => onCellEnter(e, { head: `${DAY_NAMES[d]} · ${hh(h)}–${hh((h + 1) % 24)}`, v })}
                    />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <div className="whm-scroll">
          <table className="whm-table junctions" aria-label={`${metric === "rush" ? "Rush-hour" : "All-day"} average congestion by junction`}>
            <thead>
              <tr>
                <th scope="col" className="jn">Junction</th>
                {DAYS.map((d, i) => <th key={d} scope="col" className={i === today ? "now" : ""}>{d}</th>)}
              </tr>
            </thead>
            <tbody>
              {junctionRows.map((r) => (
                <tr key={r.name}>
                  <th scope="row" className="jn">
                    <button className="whm-jlink" onClick={() => { setJunction(r.name); setMode("hours"); }} title="Show this junction by hour">
                      {r.name}
                    </button>
                  </th>
                  {r.values.map((v, d) => (
                    <td
                      key={d}
                      className={d === today ? "cell num today" : "cell num"}
                      style={{ background: rampColor(v), color: v >= 62 ? "#1f1e1d" : "var(--text)" }}
                      aria-label={`${r.name}, ${DAY_NAMES[d]}: ${v}%`}
                      onMouseEnter={(e) => onCellEnter(e, { head: `${r.name} · ${DAY_NAMES[d]}`, v, sub: metric === "rush" ? "rush-hour average" : "all-day average" })}
                    >
                      {v}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="whm-foot">
        <div className="whm-legend" aria-hidden="true">
          <span>Light</span>
          <div className="whm-legend-bar" style={{ background: `linear-gradient(90deg, ${RAMP.join(", ")})` }} />
          <span>Severe</span>
          <span className="whm-legend-scale">0% → 100% congestion</span>
          <span className="whm-now-key"><i /> now</span>
        </div>
        <span className="whm-note">
          Modelled typical week: each junction&apos;s live load shaped by Bhubaneswar weekday/weekend peak patterns.
        </span>
      </div>

      {hover && (
        <div className="whm-tip" style={{ left: hover.x, top: hover.y }} role="tooltip">
          <div className="whm-tip-head">{hover.head}</div>
          <div className="whm-tip-val">
            <span className="whm-tip-swatch" style={{ background: rampColor(hover.v) }} />
            <strong>{hover.v}%</strong> {level(hover.v)}{hover.sub ? ` · ${hover.sub}` : ""}
          </div>
        </div>
      )}
    </div>
  );
}

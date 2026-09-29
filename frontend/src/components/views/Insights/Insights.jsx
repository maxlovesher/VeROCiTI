import { useCallback, useEffect, useState } from "react";
import { applyRetentionLimit, fetchInsightsSummary, lookupPlate } from "../../../services/insightsService";
import "./Insights.css";

const REFRESH_MS = 15000;

const TYPE_LABEL = {
  SECTION_SPEED: "Section speed",
  HEAVY_VEHICLE_RESTRICTED_HOURS: "Heavy vehicle, restricted hours",
  SCHOOL_ZONE_SPEED: "School-zone speed",
  IMPOSSIBLE_TRAVEL: "Impossible travel — review",
};

const LEVEL_CLASS = { FULL_COVERAGE: "green", PARTIAL_COVERAGE: "amber", MINIMAL_COVERAGE: "red", NO_DATA: "muted" };

function ago(seconds) {
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)} h ago`;
}

function Kpi({ label, value, sub, tone }) {
  return (
    <div className="ins-kpi">
      <span className="ins-kpi-label">{label}</span>
      <span className={`ins-kpi-value ${tone || ""}`}>{value}</span>
      {sub && <span className="ins-kpi-sub">{sub}</span>}
    </div>
  );
}

function PlateLookup() {
  const [plate, setPlate] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  async function submit(e) {
    e.preventDefault();
    if (!plate.trim()) return;
    setLoading(true);
    setError(null);
    try {
      setResult(await lookupPlate(plate.trim()));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  const r = result?.result;
  return (
    <div className="ins-lookup">
      <form onSubmit={submit} className="ins-lookup-form">
        <input
          value={plate}
          onChange={(e) => setPlate(e.target.value)}
          placeholder="Plate as read, e.g. 0D02BA4455"
          aria-label="Plate to look up"
        />
        <button className="ins-btn" disabled={loading}>{loading ? "Checking…" : "Match"}</button>
      </form>
      {error && <div className="ins-error">{error}</div>}
      {r && (
        <div className="ins-lookup-result">
          <div>
            <span className={`ins-pill ${r.matched ? "green" : "muted"}`}>{r.method}</span>{" "}
            {r.matched ? <>matched <strong>{r.plate}</strong> (score {r.score})</> : "no confident match"}
            {result.watchlist_hit && <span className="ins-pill red">Watchlist hit</span>}
          </div>
          <div className="ins-hint">
            Checked against {result.known_plates} plates seen. Closest:{" "}
            {result.nearest.map((n) => `${n.plate} (${n.similarity})`).join(", ") || "none"}
          </div>
        </div>
      )}
    </div>
  );
}

export default function Insights() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [purgeMsg, setPurgeMsg] = useState(null);

  const load = useCallback(async () => {
    try {
      setData(await fetchInsightsSummary(24));
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  async function purge() {
    const pending = data?.privacy?.retention?.rows_past_retention ?? 0;
    if (!window.confirm(`Permanently delete ${pending} detection record(s) past the retention limit?`)) return;
    try {
      const r = await applyRetentionLimit();
      setPurgeMsg(`Deleted ${r.deleted} record(s); ${r.kept} kept.`);
      load();
    } catch (e) {
      setPurgeMsg(e.message);
    }
  }

  if (!data) {
    return (
      <div className="ins-view">
        <div className="ins-card ins-empty">
          {error
            ? <><i className="fas fa-plug-circle-xmark" /> Can't reach the analytics API ({error}). Start the Python backend with <code>start.bat</code>.</>
            : <><i className="fas fa-spinner fa-spin" /> Loading camera-network analytics…</>}
        </div>
      </div>
    );
  }

  const health = data.camera_health;
  const travel = data.travel_time;
  const incidents = data.incidents;
  const demand = data.demand;
  const enf = data.enforcement;
  const traj = data.trajectories;
  const priv = data.privacy;

  return (
    <div className="ins-view">
      {error && <div className="ins-error">Last refresh failed: {error}. Showing data from {data.generated_at}.</div>}

      <div className="ins-kpis">
        <Kpi label="Detections analysed" value={data.detections_analysed} sub={`last ${data.hours} h`} />
        <Kpi
          label="Camera coverage"
          value={`${health.healthy}/${health.fleet_size}`}
          sub={health.fallback_level.replace("_", " ").toLowerCase()}
          tone={LEVEL_CLASS[health.fallback_level]}
        />
        <Kpi label="Likely incidents" value={incidents.incidents.length} tone={incidents.incidents.length ? "red" : "green"} />
        <Kpi
          label="Violations"
          value={enf.total}
          sub={enf.needs_review ? `+${enf.needs_review} misread/cloned-plate reviews` : "none needing review"}
          tone={enf.total ? "amber" : ""}
        />
        <Kpi
          label="Unusual demand"
          value={demand.warming_up ? "—" : demand.alerts.length}
          sub={demand.warming_up ? `needs ${demand.minutes_needed} min of data` : `${demand.window_minutes}-min windows`}
          tone={demand.alerts.length ? "amber" : ""}
        />
      </div>

      <div className="ins-grid">
        {/* Feature 19 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-car-burst" /> Incident detection</div>
          <div className="ins-hint">A link is flagged when 3+ expected arrivals are overdue <em>and</em> arrivals downstream are rising.</div>
          {incidents.links_watched.length === 0 ? (
            <div className="ins-hint">No camera-to-camera links with enough trips to watch yet.</div>
          ) : (
            <table className="ins-table">
              <thead><tr><th>Link</th><th>Typical</th><th>Overdue</th><th>Trend</th></tr></thead>
              <tbody>
                {incidents.links_watched.slice(0, 8).map((l) => (
                  <tr key={l.link} className={l.incident_likely ? "flag" : ""}>
                    <td>{l.from_name} → {l.to_name}</td>
                    <td>{l.typical_minutes} min</td>
                    <td>{l.overdue}/{l.pending}</td>
                    <td>{l.incident_likely ? <span className="ins-pill red">Incident</span> : l.queue_trend.toLowerCase()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* Feature 20 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-video" /> Camera health</div>
          <div className="ins-hint">
            A camera is healthy if it reported in the last {Math.round(health.timeout_s / 60)} min. Coverage level:{" "}
            <span className={`ins-pill ${LEVEL_CLASS[health.fallback_level]}`}>{health.fallback_level}</span>
          </div>
          <div className="ins-cams">
            {health.cameras.map((c) => (
              <div key={c.camera_id} className={`ins-cam ${c.healthy ? "ok" : "down"}`} title={c.camera_id}>
                <span className="ins-dot" />
                <span className="ins-cam-name">{c.name}</span>
                <span className="ins-cam-age">{ago(c.seconds_since_last)}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature 22 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-stopwatch" /> Travel-time reliability</div>
          <div className="ins-hint">Median and 95th-percentile minutes between cameras. Reliability ratio ≥ 1.5 with 3+ trips is a bottleneck.</div>
          {travel.links.length === 0 ? (
            <div className="ins-hint">No repeat sightings between cameras yet.</div>
          ) : (
            <table className="ins-table">
              <thead><tr><th>Link</th><th>Median</th><th>P95</th><th>Ratio</th><th>Trips</th></tr></thead>
              <tbody>
                {travel.links.slice(0, 8).map((l) => {
                  const bottleneck = travel.bottlenecks.some((b) => b.road === l.link);
                  return (
                    <tr key={l.link} className={bottleneck ? "flag" : ""}>
                      <td>{l.from_name} → {l.to_name}</td>
                      <td>{l.median}</td>
                      <td>{l.p95}</td>
                      <td>{l.reliability_ratio ?? "—"}{bottleneck && <span className="ins-pill amber">Bottleneck</span>}</td>
                      <td>{l.sample_size}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>

        {/* Feature 23 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-people-group" /> Unusual demand</div>
          {demand.warming_up ? (
            <div className="ins-hint">
              Building a baseline: {demand.minutes_of_data} of {demand.minutes_needed} minutes of history collected.
              Alerts start once every camera has {demand.history_windows} past windows to compare against.
            </div>
          ) : demand.alerts.length === 0 ? (
            <div className="ins-hint">All {demand.cameras_tracked} cameras are within their normal range (z-score below 2.5).</div>
          ) : (
            <table className="ins-table">
              <thead><tr><th>Camera</th><th>Now</th><th>Usual</th><th>z</th></tr></thead>
              <tbody>
                {demand.alerts.map((a) => (
                  <tr key={a.camera_id} className="flag">
                    <td>{a.name}</td>
                    <td>{a.current_count}</td>
                    <td>{a.baseline_mean} ± {a.baseline_std}</td>
                    <td>{a.z_score}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* Features 12–17 */}
        <section className="ins-card wide">
          <div className="ins-title"><i className="fas fa-gavel" /> Automated enforcement</div>
          <div className="ins-rules">
            <span>Section speed limit {enf.rules.section_speed_limit_kmph} km/h (straight-line, so never over-reports)</span>
            <span>Goods vehicles ({enf.rules.heavy_vehicle_classes.join(", ")}) allowed {enf.rules.heavy_vehicle_allowed.join(", ")}</span>
            {Object.entries(enf.rules.school_zones).map(([cam, windows]) => (
              <span key={cam}>School zone {cam}: {windows.join(", ")}</span>
            ))}
            <span>Red light, wrong way and junction parking need signal/lane/dwell evidence: POST /api/insights/enforcement/check</span>
            <span>Hops needing over 140 km/h are flagged for review as a likely misread or cloned plate, never ticketed</span>
          </div>
          {enf.findings.length === 0 ? (
            <div className="ins-hint">No violations or review items in the last {data.hours} h.</div>
          ) : (
            <table className="ins-table">
              <thead><tr><th>Time</th><th>Plate</th><th>Rule</th><th>Where</th><th>Detail</th></tr></thead>
              <tbody>
                {enf.findings.slice(0, 12).map((f, i) => (
                  <tr key={`${f.plate}-${f.time}-${i}`} className={f.review ? "review" : ""}>
                    <td>{f.time.replace("T", " ").slice(0, 16)}</td>
                    <td className="mono">{f.plate}</td>
                    <td>{TYPE_LABEL[f.type] || f.type}</td>
                    <td>{f.where}</td>
                    <td>
                      {f.details.required_kmph != null && `would need ${f.details.required_kmph} km/h`}
                      {f.details.section_speed_kmph != null && `${f.details.section_speed_kmph} km/h`}
                      {f.details.speed_kmph != null && `${f.details.speed_kmph} km/h in ${f.details.limit_kmph} zone`}
                      {f.details.vehicle_class && `${f.details.vehicle_class} at ${f.details.time?.slice(0, 5)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {/* Feature 18 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-magnifying-glass" /> Fuzzy plate match</div>
          <div className="ins-hint">Tolerates OCR confusions 0/O, 8/B and 1/I; the watchlist check runs on hashed plates.</div>
          <PlateLookup />
        </section>

        {/* Feature 2 */}
        <section className="ins-card">
          <div className="ins-title"><i className="fas fa-route" /> Trajectories by hashed ID</div>
          <div className="ins-hint">{traj.identities} vehicles tracked without storing plates next to their paths.</div>
          <div className="ins-trajs">
            {traj.trajectories.slice(0, 6).map((t) => (
              <div key={t.hashed_id} className="ins-traj">
                <span className="mono">{t.hashed_id.slice(0, 16)}…</span>
                <span className="ins-traj-path">{t.recent_cameras.join(" → ")}</span>
                <span className="ins-hint">{t.sightings} sightings · {Math.round(t.dwell_seconds / 60)} min</span>
              </div>
            ))}
          </div>
        </section>

        {/* Feature 21 */}
        <section className="ins-card wide">
          <div className="ins-title"><i className="fas fa-user-shield" /> What we store</div>
          <div className="ins-privacy">
            {["identity", "raw_frames", "detection_records", "watchlist_matching"].map((k) => (
              <div key={k}>
                <strong>{k.replace("_", " ")}</strong>
                <p>{priv.what_we_store[k]}</p>
              </div>
            ))}
          </div>
          <div className="ins-row">
            <span className="ins-hint">
              {priv.retention.rows_total} records stored, {priv.retention.violation_rows} are violations.{" "}
              <strong>{priv.retention.rows_past_retention}</strong> past the retention limit.
            </span>
            <button className="ins-btn danger" onClick={purge} disabled={!priv.retention.rows_past_retention}>
              Apply retention limit
            </button>
            {purgeMsg && <span className="ins-hint">{purgeMsg}</span>}
          </div>
          {!priv.identity_secret_configured && (
            <div className="ins-warn">
              <i className="fas fa-triangle-exclamation" /> IDENTITY_HASH_SECRET isn't set, so hashes use the built-in development key. Set it on the server before real deployment.
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

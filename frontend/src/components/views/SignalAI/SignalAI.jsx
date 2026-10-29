import { useState } from "react";
import {
  dispatchAmbulance,
  standDownAmbulance,
  setControlMode,
  setSensorFault,
  setPedestrianCall,
} from "../../../services/cityFlowService";
import "./SignalAI.css";

const RUNGS = [
  { id: "FULL_AI", label: "Full AI", desc: "PCU queues, downstream spillback, incidents, live priorities" },
  { id: "HISTORICAL_PROFILE", label: "Historical profile", desc: "Time-of-day green split, no live data" },
  { id: "LOCAL_ACTUATED", label: "Local actuated", desc: "Own queues only, no neighbour data" },
  { id: "FIXED_TIME", label: "Fixed time", desc: "Plain 25 s cycle, ignores sensors" },
];
const PHASE_LABEL = { EW: "East–West", NS: "North–South" };
const TREND_ICON = { RISING: "fa-arrow-trend-up", FALLING: "fa-arrow-trend-down", STABLE: "fa-minus" };
const JUNCTIONS = ["J1", "J2", "J3", "J4", "J5"];

function signalLabel(agent, tl) {
  if (agent.is_all_red) return { text: "ALL RED", cls: "red" };
  if (agent.is_yellow) return { text: "YELLOW", cls: "amber" };
  const phase = (tl?.phase_idx ?? 0) === 0 ? "EW" : "NS";
  return { text: `${phase} GREEN`, cls: "green" };
}

function Approach({ jid, phase, obs, bus, isGreen, onPedCall, busy }) {
  if (!obs) return null;
  const adv = obs.speed_advisory || {};
  const pcuHeavier = obs.queue_length_pcu > obs.queue_length + 0.05;
  return (
    <div className={`sai-approach${isGreen ? " green" : ""}`}>
      <div className="sai-approach-head">
        <span className="sai-approach-name">{PHASE_LABEL[phase]}</span>
        <span className={`sai-light ${isGreen ? "green" : "red"}`} />
      </div>
      <div className="sai-metrics">
        <div>
          <span className="sai-k">Queue</span>
          <span className="sai-v">
            {obs.queue_length} veh
            <em className={pcuHeavier ? "heavy" : ""}> · {obs.queue_length_pcu} PCU</em>
          </span>
        </div>
        <div>
          <span className="sai-k">In 5 min</span>
          <span className="sai-v">
            {obs.predicted_queue_5min} PCU{" "}
            <i className={`fas ${TREND_ICON[obs.queue_trend] || "fa-minus"} trend-${obs.queue_trend}`} title={obs.queue_trend} />
          </span>
        </div>
        <div>
          <span className="sai-k">Red wait</span>
          <span className="sai-v">
            {Math.round(obs.red_wait_s ?? 0)} s
            {obs.aging_boost > 0 && <em className="aging" title="Fair scheduling: priority added for time spent waiting"> · +{obs.aging_boost}</em>}
          </span>
        </div>
      </div>
      <div className="sai-advisory" title="Green-light speed advisory for this approach's VMS">
        <i className="fas fa-gauge-high" /> {adv.message || "—"}
      </div>
      <div className="sai-tags">
        {bus?.late_buses > 0 && (
          <span className="sai-tag bus" title="Buses running 3+ minutes late get a bounded priority boost">
            <i className="fas fa-bus" /> {bus.late_buses} late · +{bus.boost}
          </span>
        )}
        {obs.pedestrian_waiting && (
          <span className="sai-tag ped">
            <i className="fas fa-person-walking" /> waiting {obs.pedestrian_wait_time} steps
          </span>
        )}
        {obs.pedestrian_walk_protected && (
          <span className="sai-tag ped"><i className="fas fa-shield-halved" /> walk protected</span>
        )}
        <button
          className="sai-mini-btn"
          disabled={obs.pedestrian_waiting || busy}
          onClick={() => onPedCall(jid, phase)}
          title={`Pedestrians request to cross the ${PHASE_LABEL[phase]} road`}
        >
          <i className="fas fa-hand-pointer" /> Ped call
        </button>
      </div>
    </div>
  );
}

export default function SignalAI({ rawState, isConnected }) {
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);
  const [origin, setOrigin] = useState("J1");
  const [hospital, setHospital] = useState("");
  const [severity, setSeverity] = useState(1);

  async function act(key, fn) {
    setBusy(key);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  if (!isConnected || !rawState?.control_mode) {
    return (
      <div className="sai-view">
        <div className="sai-offline">
          <i className="fas fa-plug-circle-xmark" />
          <div>
            <strong>Signal controller offline.</strong>
            <span>The adaptive controller runs in the Python backend. Start it with <code>start.bat</code> (or <code>python server_standalone.py</code>) and this page will connect.</span>
          </div>
        </div>
      </div>
    );
  }

  const mode = rawState.control_mode;
  const forced = rawState.control_mode_forced;
  const amb = rawState.ambulance || {};
  // Older servers only send the single `ambulance`; newer ones list every ambulance on the road.
  const ambulances = rawState.ambulances || (amb.active ? [amb] : []);
  const maxAmbulances = rawState.max_ambulances || 1;
  const em = rawState.emissions || {};
  const waves = rawState.green_wave || [];
  const idleMax = Math.max(em.baseline?.avg_idle_s || 0, em.live?.avg_idle_s || 0, 1);

  return (
    <div className="sai-view">
      {error && (
        <div className="sai-error"><i className="fas fa-triangle-exclamation" /> {error}</div>
      )}

      <div className="sai-top">
        {/* Feature 9 — degradation ladder */}
        <section className="sai-card">
          <div className="sai-title"><i className="fas fa-layer-group" /> Control mode</div>
          <div className="sai-ladder">
            {RUNGS.map((r) => (
              <button
                key={r.id}
                className={`sai-rung${mode === r.id ? " active" : ""}`}
                onClick={() => act(`mode-${r.id}`, () => setControlMode(r.id))}
                disabled={!!busy}
                title={`Pin the network to ${r.label}`}
              >
                <span className="sai-rung-label">{r.label}</span>
                <span className="sai-rung-desc">{r.desc}</span>
              </button>
            ))}
          </div>
          <div className="sai-row">
            <span className={`sai-pill ${forced ? "amber" : "green"}`}>
              {forced ? "Pinned by operator" : "Automatic (follows data health)"}
            </span>
            {forced && (
              <button className="sai-btn" disabled={!!busy} onClick={() => act("auto", () => setControlMode("AUTO"))}>
                Release to auto
              </button>
            )}
          </div>
          <label className="sai-toggle">
            <input
              type="checkbox"
              checked={!!rawState.sensor_fault}
              disabled={!!busy}
              onChange={(e) => act("fault", () => setSensorFault(e.target.checked))}
            />
            Simulate vehicle-feed failure
            <span className="sai-hint">steps down a rung every 3 bad readings, climbs back after 10 good ones</span>
          </label>
        </section>

        {/* Feature 11 — ambulance corridors, several at once */}
        <section className={`sai-card${ambulances.length ? " alert" : ""}`}>
          <div className="sai-title">
            <i className="fas fa-truck-medical" /> Ambulance corridors
            {ambulances.length > 0 && <span className="sai-count">{ambulances.length}/{maxAmbulances}</span>}
          </div>
          {ambulances.map((a) => (
            <div key={a.id || "amb"} className="sai-amb">
              <div className="sai-amb-status">
                <strong>{a.id}</strong> → <strong>{a.hospital}</strong> · {Math.round(a.progress_m)} m
                {a.severity > 1 && <span className="sai-tag sev">severity {a.severity}</span>}
                {a.waiting_at && <span className="sai-tag wait">stopped at {a.waiting_at}</span>}
              </div>
              <div className="sai-route">
                {(a.route_junctions || []).map((jid, i, r) => {
                  const cleared = a.cleared_junctions?.includes(jid);
                  const preempted = a.triggered_junctions?.includes(jid) && !cleared;
                  const yieldTo = a.yielding_to?.[jid];
                  const isHospital = i === r.length - 1;
                  const state = isHospital ? "hospital" : cleared ? "cleared" : preempted ? "preempted" : yieldTo ? "yielding" : "pending";
                  const label = isHospital ? "hospital" : cleared ? "cleared" : preempted ? "green held"
                    : yieldTo ? `after ${yieldTo}` : `ETA ${a.eta_by_junction?.[jid] ?? "–"}`;
                  return (
                    <div key={jid} className={`sai-stop ${state}`}>
                      <span className="sai-stop-id">{jid}</span>
                      <span className="sai-stop-state">{label}</span>
                    </div>
                  );
                })}
              </div>
              <button className="sai-mini-btn" disabled={!!busy} onClick={() => act(`amb-${a.id}`, () => standDownAmbulance(a.id))}>
                Stand down {a.id}
              </button>
            </div>
          ))}
          {ambulances.length === 0 && (
            <div className="sai-amb-status">
              Routes to the nearest hospital, pre-empts each junction by live ETA (earlier when a queue is waiting),
              runs amber + all-red before the emergency green, then serves the most-starved approach first.
              When two ambulances need one junction in different directions, the one arriving first goes first.
            </div>
          )}
          {ambulances.length < maxAmbulances && (
            <div className="sai-row">
              <select value={origin} onChange={(e) => setOrigin(e.target.value)} className="sai-select">
                {JUNCTIONS.map((j) => <option key={j} value={j}>From {j}</option>)}
              </select>
              <select value={hospital} onChange={(e) => setHospital(e.target.value)} className="sai-select">
                <option value="">Nearest hospital</option>
                {Object.keys(rawState.hospitals || {}).map((h) => <option key={h} value={h}>{h}</option>)}
              </select>
              <select value={severity} onChange={(e) => setSeverity(Number(e.target.value))} className="sai-select">
                <option value={1}>Routine</option>
                <option value={2}>Urgent</option>
                <option value={3}>Critical</option>
              </select>
              <button
                className="sai-btn primary"
                disabled={!!busy}
                onClick={() => act("amb", () => dispatchAmbulance(origin, { hospital, severity }))}
              >
                <i className="fas fa-play" /> {ambulances.length ? "Dispatch another" : "Dispatch"}
              </button>
            </div>
          )}
          {ambulances.length > 1 && (
            <button className="sai-btn danger" disabled={!!busy} onClick={() => act("amb-all", () => standDownAmbulance())}>
              Stand down all
            </button>
          )}
          <div className="sai-hint">
            Hospitals: {Object.entries(rawState.hospitals || {}).map(([h, j]) => `${h} (${j})`).join(", ")}
          </div>
        </section>

        {/* Feature 24 — emissions vs fixed time */}
        <section className="sai-card">
          <div className="sai-title"><i className="fas fa-leaf" /> Idle time &amp; CO₂ vs fixed-time</div>
          {em.ready ? (
            <>
              <div className="sai-bars">
                <div className="sai-bar-row">
                  <span>Adaptive</span>
                  <div className="sai-bar"><div className="fill green" style={{ width: `${(em.live.avg_idle_s / idleMax) * 100}%` }} /></div>
                  <strong>{em.live.avg_idle_s}s</strong>
                </div>
                <div className="sai-bar-row">
                  <span>Fixed time</span>
                  <div className="sai-bar"><div className="fill muted" style={{ width: `${(em.baseline.avg_idle_s / idleMax) * 100}%` }} /></div>
                  <strong>{em.baseline.avg_idle_s}s</strong>
                </div>
              </div>
              <div className="sai-big">
                <span className={em.co2_saved_kg >= 0 ? "green" : "red"}>{em.co2_saved_kg_per_100_trips} kg</span>
                <small>CO₂ saved per 100 trips · {em.idle_seconds_saved_per_trip}s less idling each</small>
              </div>
              <div className="sai-hint">Average idle per trip over the last {em.live.trips} trips. {em.note}</div>
            </>
          ) : (
            <div className="sai-hint">Collecting trips for both simulations… ({em.live?.trips ?? 0} adaptive, {em.baseline?.trips ?? 0} fixed-time so far)</div>
          )}
        </section>

        {/* Feature 5 — green waves */}
        <section className="sai-card">
          <div className="sai-title"><i className="fas fa-wave-square" /> Green-wave corridors</div>
          {waves.length === 0 ? (
            <div className="sai-hint">Waiting for origin–destination trips…</div>
          ) : (
            waves.map((w) => (
              <div key={w.junctions.join("-")} className="sai-wave">
                <div className="sai-wave-chain">
                  {w.junctions.map((j, i) => (
                    <span key={j} className="sai-wave-node">
                      {j}
                      <small>+{w.offsets_s[i]}s</small>
                    </span>
                  ))}
                </div>
                <div className="sai-hint">{w.trips} trips · {w.phase_sequence.join(" → ")} phases · offsets at 40 km/h</div>
              </div>
            ))
          )}
        </section>
      </div>

      {/* Per-junction controller state (Features 1, 3–4, 6–8, 10–11) */}
      <div className="sai-junctions">
        {JUNCTIONS.map((jid) => {
          const agent = rawState.agents?.[jid];
          if (!agent) return null;
          const tl = rawState.tl_phases?.[jid];
          const sig = signalLabel(agent, tl);
          const greenPhase = agent.is_yellow || agent.is_all_red ? null : ((tl?.phase_idx ?? 0) === 0 ? "EW" : "NS");
          return (
            <section key={jid} className={`sai-card sai-junction${agent.emergency_active ? " alert" : ""}`}>
              <div className="sai-jhead">
                <span className="sai-jid">{jid}</span>
                <span className={`sai-pill ${sig.cls}`}>{sig.text}</span>
                <span className="sai-jtime">{agent.steps_on_phase}/{agent.allocated_green} steps</span>
                {agent.emergency_active && <span className="sai-pill red"><i className="fas fa-truck-medical" /> Pre-empted</span>}
                {agent.recovery_mode && <span className="sai-pill amber">Recovery</span>}
                {agent.control_mode !== "FULL_AI" && <span className="sai-pill muted">{agent.control_mode}</span>}
              </div>
              <div className="sai-reason">{agent.decision_reason}</div>
              {["EW", "NS"].map((p) => (
                <Approach
                  key={p}
                  jid={jid}
                  phase={p}
                  obs={agent.local_obs?.[p]}
                  bus={rawState.bus_priority?.[jid]?.[p]}
                  isGreen={greenPhase === p}
                  busy={!!busy}
                  onPedCall={(j, ph) => act(`ped-${j}-${ph}`, () => setPedestrianCall(j, ph, true))}
                />
              ))}
            </section>
          );
        })}
      </div>
    </div>
  );
}

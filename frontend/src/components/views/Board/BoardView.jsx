import { useCallback, useEffect, useState } from "react";
import {
  fetchBoardState,
  listPorts,
  setBoardMode,
  setManualLamp,
  setWiring,
  setExchanged,
  setPort,
} from "../../../services/boardService";
import CameraPanel from "./CameraPanel";
import "./BoardView.css";

const MODES = [
  { id: "follow", label: "Follow AI", icon: "fa-brain", hint: "The lights on the board mirror the Signal AI running in the simulation." },
  { id: "camera", label: "AI from camera", icon: "fa-video", hint: "The Signal AI decides from the cars the camera sees. A pink AMBULANCE card opens a green corridor. If the camera goes quiet it steps down to a fixed-time plan on its own." },
  { id: "test", label: "Lamp test", icon: "fa-lightbulb", hint: "Every head pair in turn, one lamp at a time. A lamp that lights out of order is mis-wired." },
  { id: "allon", label: "All on", icon: "fa-sun", hint: "Every lamp on at once: all 24 LEDs. Keep it short; a dark LED is a loose wire or a dead module." },
  { id: "wiring", label: "Wiring check", icon: "fa-plug", hint: "" },
  { id: "manual", label: "Manual", icon: "fa-hand-pointer", hint: "Set each head pair yourself. Choose Follow AI to hand control back." },
];
const LAMPS = ["R", "Y", "G"];
const MANUAL_CHOICES = [
  { id: "R", label: "Red" },
  { id: "Y", label: "Amber" },
  { id: "G", label: "Green" },
  { id: "O", label: "Off" },
  { id: "W", label: "All" },
];
const DIRECTIONS = [
  { id: "EW", label: "East–West" },
  { id: "NS", label: "North–South" },
];

function summary(sig) {
  const ew = sig?.EW ?? "O";
  const ns = sig?.NS ?? "O";
  if (ew === "W" || ns === "W") return "All lamps on";
  if (ew === "O" && ns === "O") return "Off";
  if (ew === "G") return "East–West green";
  if (ns === "G") return "North–South green";
  if (ew === "Y" || ns === "Y") return "Amber";
  return "All red";
}

function Head({ state, name }) {
  const lit = (c) => state === "W" || state === c;
  return (
    <div className="lb-head" title={`${name} head`}>
      {LAMPS.map((c) => <i key={c} className={`lb-lamp ${c.toLowerCase()}${lit(c) ? " on" : ""}`} />)}
    </div>
  );
}

function Junction({ id, sig, manual, swapped, isManual, busy, onLamp, onSwap }) {
  return (
    <section className="lb-card">
      <header className="lb-card-head">
        <h2>{id}</h2>
        <span className="lb-phase">{summary(sig)}</span>
      </header>

      <div className="lb-cross" role="img" aria-label={`${id}: ${summary(sig)}`}>
        <div className="lb-cell n"><Head state={sig?.NS} name="North" /></div>
        <div className="lb-cell w"><Head state={sig?.EW} name="West" /></div>
        <div className="lb-road">{id}</div>
        <div className="lb-cell e"><Head state={sig?.EW} name="East" /></div>
        <div className="lb-cell s"><Head state={sig?.NS} name="South" /></div>
      </div>

      {isManual && (
        <div className="lb-manual">
          {DIRECTIONS.map((d) => (
            <div key={d.id} className="lb-manual-row">
              <span>{d.label}</span>
              <div className="lb-seg">
                {MANUAL_CHOICES.map((c) => (
                  <button key={c.id} disabled={busy} className={manual?.[d.id] === c.id ? "active" : ""}
                    onClick={() => onLamp(id, d.id, c.id)}>{c.label}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <footer className="lb-card-foot">
        <label className="lb-switch" title="Turn on if this junction's lamps light the wrong pair">
          <input type="checkbox" checked={swapped} disabled={busy} onChange={(e) => onSwap(id, e.target.checked)} />
          <span className="lb-track"><span className="lb-thumb" /></span>
          Pairs wired crossed
        </label>
      </footer>
    </section>
  );
}

export default function BoardView() {
  const [state, setState] = useState(null);
  const [offline, setOffline] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [ports, setPorts] = useState([]);

  const refresh = useCallback(async () => {
    try {
      setState(await fetchBoardState());
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  const loadPorts = useCallback(() => {
    listPorts().then((r) => setPorts(r.ports || [])).catch(() => setPorts([]));
  }, []);

  useEffect(() => {
    refresh();
    loadPorts();
    const id = setInterval(refresh, 600);
    return () => clearInterval(id);
  }, [refresh, loadPorts]);

  const act = async (fn) => {
    setBusy(true);
    setError("");
    try {
      setState(await fn());
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  if (offline && !state) {
    return (
      <div className="lb-view">
        <div className="lb-empty">
          <i className="fas fa-plug-circle-xmark" />
          <div>
            <strong>The board service isn't reachable</strong>
            <span>Start the backend with <code>start.bat</code>, then this page connects by itself.</span>
          </div>
        </div>
      </div>
    );
  }
  if (!state) return <div className="lb-view"><div className="lb-note">Connecting…</div></div>;

  const link = state.link || {};
  const mode = MODES.find((m) => m.id === state.mode) || MODES[0];
  const modeHint = state.mode === "wiring" ? state.wiring_hint : mode.hint;
  const connected = link.connected;
  const setting = link.setting || "auto";

  return (
    <div className="lb-view">
      <div className="lb-bar">
        <div className={`lb-conn ${connected ? "ok" : "bad"}`}>
          <i className="lb-dot" />
          <div>
            <strong>{connected ? `Arduino Uno on ${link.port}` : setting === "off" ? "Port released" : "Not connected"}</strong>
            <span>{connected ? `${link.frames_sent} updates sent` : link.error || "Looking for the Uno…"}</span>
          </div>
        </div>
        <div className="lb-modes" role="radiogroup" aria-label="What drives the lights">
          {MODES.map((m) => (
            <button key={m.id} role="radio" aria-checked={state.mode === m.id} disabled={busy}
              className={state.mode === m.id ? "active" : ""} onClick={() => act(() => setBoardMode(m.id))}>
              <i className={`fas ${m.icon}`} /> {m.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="lb-error">{error}</div>}
      {state.sim_error && (state.mode === "follow" || state.mode === "camera") && (
        <div className="lb-error">The Signal AI isn't answering ({state.sim_error}). The board flashes amber until it does.</div>
      )}
      {modeHint && <p className={`lb-hint${state.mode === "wiring" ? " strong" : ""}`}>{modeHint}</p>}

      <div className="lb-grid">
        {(state.junctions || []).map((id) => (
          <Junction key={id} id={id} sig={state.signals?.[id]} manual={state.manual?.[id]}
            swapped={(state.swapped || []).includes(id)} isManual={state.mode === "manual"} busy={busy}
            onLamp={(j, d, c) => act(() => setManualLamp(j, d, c))}
            onSwap={(j, v) => act(() => setWiring(j, v))} />
        ))}
      </div>

      <CameraPanel camera={state.camera} mode={state.mode} act={act} busy={busy} />

      <section className="lb-card">
        <header className="lb-card-head"><h2>Connection</h2></header>
        <div className="lb-conn-row">
          <label htmlFor="lb-port">USB port</label>
          <select id="lb-port" value={setting} disabled={busy} onFocus={loadPorts}
            onChange={(e) => act(() => setPort(e.target.value))}>
            <option value="auto">Auto-detect the Arduino</option>
            <option value="off">Off (let go of the port)</option>
            {ports.map((p) => (
              <option key={p.port} value={p.port}>{p.port}{p.description ? ` · ${p.description}` : ""}</option>
            ))}
          </select>
        </div>
        <div className="lb-conn-row">
          <label>Junction pins</label>
          <label className="lb-switch" title="Turn on if, in the Wiring check, the left junction shows what the right one should">
            <input type="checkbox" checked={!!state.exchanged} disabled={busy} onChange={(e) => act(() => setExchanged(e.target.checked))} />
            <span className="lb-track"><span className="lb-thumb" /></span>
            Left and right junction are on each other's pins
          </label>
        </div>
        <div className="lb-conn-row">
          <label>Sent to the Uno</label>
          <code className="lb-frame">{state.frame || "—"}</code>
        </div>
        <p className="lb-hint">One letter per head pair: J1 west+east, J1 north+south, J2…, J3 west+east, J3 north+south, … R red, Y amber, G green, O off, W all on.</p>
      </section>
    </div>
  );
}

import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchBoardState,
  listSerialPorts,
  setBoardCamera,
  calibrateBoard,
  autoCalibrateBoard,
  resetCalibration,
  captureEmptyBoard,
  clearEmptyBoard,
  setBoardMode,
  setBoardLayout,
  setJunctionWiring,
  setRegisteredPlates,
  setSerialPort,
  resetBoardAI,
  setManualLight,
  revertManualControl,
  toggleSyntheticCar,
  clearSyntheticCars,
  boardFrameUrl,
} from "../../../services/boardService";
import "./BoardView.css";

const MODES = [
  { id: "camera", label: "AI from camera", icon: "fa-brain" },
  { id: "sim", label: "Mirror simulation", icon: "fa-microchip" },
  { id: "test", label: "Lamp test", icon: "fa-lightbulb" },
  { id: "manual", label: "Manual", icon: "fa-hand-pointer" },
];
const MODE_LABEL = Object.fromEntries(MODES.map((m) => [m.id, m.label]));
const LAMPS = [
  { id: "R", label: "Red", cls: "r" },
  { id: "Y", label: "Yellow", cls: "y" },
  { id: "G", label: "Green", cls: "g" },
  { id: "O", label: "Off", cls: "o" },
];
// The four heads of a junction; each pair shares the same Nano pins, so they always light together.
const HEADS = [
  { id: "N", dir: "NS", label: "North" },
  { id: "W", dir: "EW", label: "West" },
  { id: "E", dir: "EW", label: "East" },
  { id: "S", dir: "NS", label: "South" },
];
// Where each junction sits in the plus, on a 3x3 grid (row, column).
const PLUS = { J2: [1, 2], J1: [2, 1], J3: [2, 2], J4: [2, 3], J5: [3, 2] };
const LAYOUT_LABEL = { single: "One junction (J1)", double: "Two junctions (J1 + J3)", full: "Five junctions" };
const PHASE_LABEL = { EW: "East–West", NS: "North–South" };
const CAMERA_PRESETS = [
  { label: "Phone (IP Webcam)", value: "http://192.168.43.1:8080/video" },
  { label: "USB webcam", value: "0" },
  { label: "Synthetic board", value: "synthetic" },
];

function Head({ state }) {
  return (
    <span className="bd-head" aria-label={state}>
      <i className={state === "R" ? "on r" : ""} />
      <i className={state === "Y" ? "on y" : ""} />
      <i className={state === "G" ? "on g" : ""} />
    </span>
  );
}

function Junction({ jid, j, ambulance, solo }) {
  const [row, col] = PLUS[jid];
  const pre = ambulance?.[jid];
  return (
    <div className={`bd-junction${pre ? " emergency" : ""}${solo ? " solo" : ""}`} style={solo ? undefined : { gridRow: row, gridColumn: col }}>
      <div className="bd-junction-head">
        <strong>{jid}</strong>
        {pre && <span className="bd-pill red"><i className="fas fa-truck-medical" /> {pre}</span>}
        {!pre && j.recovery && <span className="bd-pill amber">Recovery</span>}
      </div>
      {["EW", "NS"].map((p) => (
        <div key={p} className="bd-phase">
          <Head state={j.signals?.[p]} />
          <span className="bd-phase-name">{PHASE_LABEL[p]}</span>
          <span className="bd-cars">{j.cars?.[p] ?? 0} <i className="fas fa-car-side" /></span>
        </div>
      ))}
      <div className="bd-reason" title={j.reason}>{j.reason}</div>
    </div>
  );
}

function StatusChip({ ok, warn, icon, label, detail }) {
  const cls = ok ? "green" : warn ? "amber" : "red";
  return (
    <div className={`bd-chip ${cls}`}>
      <i className={`fas ${icon}`} />
      <div>
        <span className="bd-chip-label">{label}</span>
        <span className="bd-chip-detail">{detail}</span>
      </div>
    </div>
  );
}

function ManualControl({ state, busy, act, active }) {
  const [junction, setJunction] = useState(active[0] || "J1");
  const [dir, setDir] = useState("NS");
  const manual = state.mode === "manual";
  const jid = active.includes(junction) ? junction : active[0];
  const lights = state.signals?.[jid] || {};
  const go = (s) => s === "G" || s === "Y";
  const conflict = go(lights.EW) && go(lights.NS);
  const set = (j, d, s) => act(() => setManualLight(j, d, s));

  return (
    <div className={`bd-card bd-manual${manual ? " on" : ""}`}>
      <div className="bd-title">
        <i className="fas fa-hand-pointer" /> Manual control
        {manual ? (
          <span className="bd-pill amber">Manual — AI paused</span>
        ) : (
          <span className="bd-pill muted">Off — {MODE_LABEL[state.mode] || state.mode} is driving the LEDs</span>
        )}
        <span className="bd-manual-actions">
          {manual ? (
            <button className="bd-btn primary" disabled={busy} onClick={() => act(revertManualControl)}>
              <i className="fas fa-rotate-left" /> Revert to {MODE_LABEL[state.manual?.previous_mode] || "automatic"}
            </button>
          ) : (
            <button className="bd-btn primary" disabled={busy} onClick={() => act(() => setBoardMode("manual"))}>
              <i className="fas fa-hand-pointer" /> Take manual control
            </button>
          )}
        </span>
      </div>

      <div className={`bd-manual-body${manual ? "" : " locked"}`}>
        <div className="bd-manual-step">
          <span className="bd-step-label"><b>1</b> Junction</span>
          <div className="bd-row">
            {active.map((j) => (
              <button key={j} className={`bd-jbtn${j === jid ? " active" : ""}`} disabled={!manual || busy} onClick={() => setJunction(j)}>
                {j}
                <span className="bd-jbtn-heads">
                  <Head state={state.signals?.[j]?.EW} />
                  <Head state={state.signals?.[j]?.NS} />
                </span>
              </button>
            ))}
          </div>
        </div>

        <div className="bd-manual-step">
          <span className="bd-step-label"><b>2</b> Light</span>
          <div className="bd-cross" role="radiogroup" aria-label={`Lights of ${jid}`}>
            <div className="bd-cross-road h" />
            <div className="bd-cross-road v" />
            <span className="bd-cross-id">{jid}</span>
            {HEADS.map((h) => (
              <button
                key={h.id}
                role="radio"
                aria-checked={dir === h.dir}
                className={`bd-xhead ${h.id}${dir === h.dir ? " selected" : ""}`}
                disabled={!manual || busy}
                onClick={() => setDir(h.dir)}
                title={`${h.label} (moves with ${h.dir === "NS" ? (h.id === "N" ? "South" : "North") : (h.id === "E" ? "West" : "East")})`}
              >
                <Head state={lights[h.dir]} />
                <span>{h.label}</span>
              </button>
            ))}
          </div>
          <span className="bd-hint">{dir === "NS" ? "North + South" : "East + West"} selected — each pair shares its wiring, so they light together.</span>
        </div>

        <div className="bd-manual-step">
          <span className="bd-step-label"><b>3</b> Colour for {jid} {dir === "NS" ? "North–South" : "East–West"}</span>
          <div className="bd-lamp-btns">
            {LAMPS.map((l) => (
              <button
                key={l.id}
                className={`bd-lamp ${l.cls}${lights[dir] === l.id ? " current" : ""}`}
                disabled={!manual || busy}
                onClick={() => set(jid, dir, l.id)}
              >
                <i />{l.label}
              </button>
            ))}
          </div>
          {conflict && (
            <div className="bd-warn"><i className="fas fa-triangle-exclamation" /> Both directions at {jid} show go — on a real road that would be a crash risk.</div>
          )}
          <span className="bd-step-label sub">Quick</span>
          <div className="bd-row">
            <button className="bd-btn small" disabled={!manual || busy} onClick={() => set(jid, "*", "R")}>{jid}: all red</button>
            {LAMPS.map((l) => (
              <button key={l.id} className="bd-btn small" disabled={!manual || busy} onClick={() => set("*", "*", l.id)}>
                Whole board: {l.label.toLowerCase()}
              </button>
            ))}
          </div>
        </div>
      </div>
      {!manual && <div className="bd-hint">Press <b>Take manual control</b> to set the lights yourself; <b>Revert</b> gives them back.</div>}
    </div>
  );
}

export default function BoardView() {
  const [state, setState] = useState(null);
  const [offline, setOffline] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);
  // false, "auto" (one click in the middle of each junction box) or "manual" (four box corners).
  const [calibrating, setCalibrating] = useState(false);
  const [points, setPoints] = useState([]);
  const [calNote, setCalNote] = useState("");
  const [cameraInput, setCameraInput] = useState(null);
  const [ports, setPorts] = useState([]);
  const [portChoice, setPortChoice] = useState(null);
  const [dropKind, setDropKind] = useState("car");
  const [platesInput, setPlatesInput] = useState(null);
  const imgRef = useRef(null);

  const refresh = useCallback(async () => {
    try {
      const s = await fetchBoardState();
      setState(s);
      setOffline(false);
      // Seed the setup inputs from the server the first time only.
      setCameraInput((cur) => (cur === null ? s.camera?.source || "" : cur));
      setPortChoice((cur) => (cur === null ? s.serial?.setting || "auto" : cur));
      setPlatesInput((cur) => (cur === null ? (s.registered_plates || []).join(", ") : cur));
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 500);
    return () => clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    const id = setInterval(() => {
      if (document.visibilityState === "visible") setTick((t) => t + 1);
    }, calibrating ? 700 : 200);
    return () => clearInterval(id);
  }, [calibrating]);

  const loadPorts = useCallback(() => {
    listSerialPorts().then((r) => setPorts(r.ports || [])).catch(() => setPorts([]));
  }, []);
  useEffect(() => { loadPorts(); }, [loadPorts]);

  const act = async (fn) => {
    setBusy(true);
    setError("");
    try {
      const res = await fn();
      if (res?.state) setState(res.state);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const onFeedClick = (e) => {
    const img = imgRef.current;
    if (!img || !img.naturalWidth) return;
    const rect = img.getBoundingClientRect();
    const u = (e.clientX - rect.left) / rect.width;
    const v = (e.clientY - rect.top) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return;
    if (calibrating) {
      const next = [...points, [Math.round(u * img.naturalWidth), Math.round(v * img.naturalHeight)]];
      const needed = calibrating === "auto" ? (state?.calibration?.auto_labels || []).length || 1 : 4;
      if (next.length < needed) {
        setPoints(next);
        return;
      }
      const mode = calibrating;
      setPoints([]);
      setCalibrating(false);
      if (mode === "auto") {
        setCalNote("Calibrating: lights off while the junction edges are measured (about 8 s)…");
        act(async () => {
          try {
            const res = await autoCalibrateBoard(next);
            setCalNote(`Calibrated: fits within ${res.result.fit_error_units} units` +
              (res.result.empty_board_captured ? ", empty board captured." : "."));
            return res;
          } catch (err) {
            setCalNote("");
            throw err;
          }
        });
      } else {
        setCalNote("");
        act(() => calibrateBoard(next));
      }
      return;
    }
    if (state?.camera?.source === "synthetic") act(() => toggleSyntheticCar(u, v, dropKind));
  };

  if (offline && !state) {
    return (
      <div className="bd-view">
        <div className="bd-offline">
          <i className="fas fa-plug-circle-xmark" />
          <div>
            <strong>Board controller not reachable</strong>
            <span>Start the backend with <code>start.bat</code> (or <code>python "city flow model/server_standalone.py"</code>).</span>
          </div>
        </div>
      </div>
    );
  }
  if (!state) return <div className="bd-view"><div className="bd-hint">Connecting to the board…</div></div>;

  const cam = state.camera || {};
  const cal = state.calibration || {};
  const ser = state.serial || {};
  const synthetic = cam.source === "synthetic";
  const view = calibrating || !cal.calibrated ? "raw" : "board";
  const cars = (state.vehicles || []).filter((v) => v.kind === "car");
  const serialErrors = Object.values(ser.errors || {});
  const nanoCount = (ser.ports || []).length;
  const active = state.active_junctions || Object.keys(PLUS);
  const solo = active.length === 1;

  return (
    <div className="bd-view">
      <div className="bd-status">
        <StatusChip ok={cam.live} icon="fa-video" label="Camera"
          detail={cam.live ? `${cam.fps || "…"} fps` : cam.source ? (cam.error || "No frames") : "Not set"} />
        <StatusChip ok={cal.calibrated && !cal.stale} warn={cal.calibrated && (!cal.reference || cal.stale)} icon="fa-crop-simple" label="Calibration"
          detail={!cal.calibrated ? "Not calibrated" :
            cal.stale ? "Camera changed: recalibrate" :
            (cal.realigned_s_ago != null && cal.realigned_s_ago < 30) ? "Re-aligned to the moved camera" :
            `${cal.reference ? "Locked + empty board" : "Locked, capture empty board"}${cal.tracking ? " · following camera" : ""}`} />
        <StatusChip ok={ser.connected} warn={ser.setting === "off"} icon="fa-microchip" label="LED boards"
          detail={ser.connected ? `${nanoCount} Nano${nanoCount > 1 ? "s" : ""} · ${ser.ports.join(", ")}` : ser.setting === "off" ? "Off" : (serialErrors[0] || "Searching…")} />
        <StatusChip ok={state.control_mode === "FULL_AI"} warn={state.control_mode !== "FIXED_TIME"} icon="fa-brain" label="AI rung"
          detail={state.mode === "camera" ? state.control_mode.replace("_", " ") : "Not driving the board"} />
        <div className="bd-modes" role="radiogroup" aria-label="What drives the LEDs">
          {MODES.map((m) => (
            <button key={m.id} role="radio" aria-checked={state.mode === m.id}
              className={`bd-mode${state.mode === m.id ? " active" : ""}`}
              disabled={busy} onClick={() => act(() => setBoardMode(m.id))}>
              <i className={`fas ${m.icon}`} /> {m.label}
            </button>
          ))}
        </div>
      </div>

      {error && <div className="bd-error">{error}</div>}

      <ManualControl state={state} busy={busy} act={act} active={active} />

      <div className="bd-main">
        <div className="bd-card">
          <div className="bd-title">
            <i className="fas fa-eye" />
            {calibrating === "auto" ? `Click the ${cal.auto_labels?.[points.length] || "junction box"} (${points.length + 1}/${(cal.auto_labels || []).length || 1})` :
              calibrating ? `Click the ${cal.labels?.[points.length] || "corner"} (${points.length + 1}/4)` :
              view === "raw" ? "Camera feed — needs calibration" : "What the AI sees"}
            {state.ambulance?.active && <span className="bd-pill red"><i className="fas fa-truck-medical" /> Ambulance corridor</span>}
          </div>
          <div className={`bd-feed${calibrating || synthetic ? " clickable" : ""}`} onClick={onFeedClick}>
            {cam.source ? (
              // The dots live in the same box as the picture, so their % positions line up with
              // the image itself rather than the wider black frame around it.
              <div className="bd-feed-frame">
                <img ref={imgRef} src={boardFrameUrl(view, tick)} alt="Live board camera" draggable={false}
                  onError={(e) => { e.currentTarget.style.visibility = "hidden"; }}
                  onLoad={(e) => { e.currentTarget.style.visibility = "visible"; }} />
                {calibrating && imgRef.current?.naturalWidth > 0 && points.map((p, i) => (
                  <span key={i} className="bd-calib-dot"
                    style={{ left: `${(p[0] / imgRef.current.naturalWidth) * 100}%`, top: `${(p[1] / imgRef.current.naturalHeight) * 100}%` }}>
                    {i + 1}
                  </span>
                ))}
              </div>
            ) : (
              <div className="bd-feed-empty">Set a camera source below to start.</div>
            )}
          </div>
          <div className="bd-row">
            {!calibrating ? (
              <>
                <button className="bd-btn primary" disabled={!cam.live || busy} onClick={() => { setPoints([]); setCalibrating("auto"); }}
                  title="Click the middle of each junction box; the exact edges are found automatically">
                  <i className="fas fa-crop-simple" /> {cal.calibrated ? "Recalibrate" : "Calibrate"}
                </button>
                <button className="bd-btn small" disabled={!cam.live || busy} onClick={() => { setPoints([]); setCalibrating("manual"); }}
                  title="Fallback: click four junction-box corners yourself">
                  Manual (4 corners)
                </button>
              </>
            ) : (
              <button className="bd-btn" onClick={() => { setCalibrating(false); setPoints([]); }}>Cancel</button>
            )}
            <button className="bd-btn" disabled={!cam.live || !cal.calibrated || busy} onClick={() => act(captureEmptyBoard)}
              title="Take this with no cars on the board. Detection then ignores lane paint, glare and LEDs.">
              <i className="fas fa-camera" /> Capture empty board
            </button>
            {cal.reference && <button className="bd-btn" disabled={busy} onClick={() => act(clearEmptyBoard)}>Clear reference</button>}
            {cal.calibrated && !calibrating && <button className="bd-btn" disabled={busy} onClick={() => act(resetCalibration)}>Reset calibration</button>}
          </div>
          {calibrating === "auto" && (
            <div className="bd-hint">
              Click inside the dark square where the roads cross, for each junction in turn. The lights switch off for a
              few seconds while the edges are measured, then the empty board is re-captured, so keep cars off the board.
            </div>
          )}
          {calibrating === "manual" && (
            <div className="bd-hint">{cal.hint}</div>
          )}
          {busy && calNote && <div className="bd-hint"><i className="fas fa-spinner fa-spin" /> {calNote}</div>}
          {!busy && calNote && !error && <div className="bd-hint"><i className="fas fa-check" /> {calNote}</div>}
          {synthetic && !calibrating && (
            <div className="bd-row bd-hint">
              Synthetic board: click a road to drop or remove a
              <select className="bd-select" value={dropKind} onChange={(e) => setDropKind(e.target.value)}>
                <option value="car">car</option>
                <option value="ambulance">ambulance</option>
              </select>
              <button className="bd-btn" disabled={busy} onClick={() => act(clearSyntheticCars)}>Clear cars</button>
            </div>
          )}
        </div>

        <div className="bd-card">
          <div className="bd-title"><i className="fas fa-traffic-light" /> Signal AI output → LEDs</div>
          <div className={solo ? "bd-solo" : active.length === 2 ? "bd-duo" : "bd-plus"}>
            {active.map((jid) => state.junctions?.[jid] && (
              <Junction key={jid} jid={jid} j={state.junctions[jid]} ambulance={state.ambulance?.junctions} solo={active.length <= 2} />
            ))}
          </div>
          <div className="bd-frame" title="The exact line sent to the Nanos every tick">
            <span>Sent to Nanos</span>
            <code>{ser.last_frame || "—"}</code>
          </div>
        </div>
      </div>

      <div className="bd-bottom">
        <div className="bd-card">
          <div className="bd-title"><i className="fas fa-id-card" /> Vehicles on the board ({cars.length})</div>
          {cars.length === 0 ? (
            <div className="bd-hint">No cars detected. Put a white box on a road.</div>
          ) : (
            <div className="bd-table-wrap">
              <table className="bd-table">
                <thead><tr><th>Plate</th><th>State</th><th>Queued at</th><th>Approach</th><th>Road</th></tr></thead>
                <tbody>
                  {cars.map((v) => (
                    <tr key={v.id}>
                      <td>
                        <strong>{v.plate || "reading…"}</strong>{v.count > 1 && <em> ×{v.count}</em>}
                        {v.registered && <i className="fas fa-circle-check bd-reg" title="Registered plate" />}
                      </td>
                      <td>{v.plate_state || "—"}</td>
                      <td>{v.in_junction ? `crossing ${v.junction}` : v.junction || "off-road"}</td>
                      <td>{v.phase || "—"}</td>
                      <td className="bd-mono">{v.road || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {state.ocr === false && <div className="bd-hint">Plate reading unavailable (EasyOCR failed to load); counting still works.</div>}
        </div>

        <div className="bd-card">
          <div className="bd-title"><i className="fas fa-sliders" /> Setup</div>
          <label className="bd-field">
            <span>Board layout</span>
            <div className="bd-row">
              <select className="bd-select" value={state.layout} disabled={busy}
                onChange={(e) => act(() => setBoardLayout(e.target.value))}>
                {(state.layouts || ["single", "full"]).map((l) => <option key={l} value={l}>{LAYOUT_LABEL[l] || l}</option>)}
              </select>
              <span className="bd-hint">Changing it clears the calibration.</span>
            </div>
          </label>
          <div className="bd-field">
            <span>Head wiring</span>
            <div className="bd-row">
              {(state.active_junctions || []).map((jid) => {
                const swapped = (state.swapped_junctions || []).includes(jid);
                return (
                  <button key={jid} className={`bd-btn small${swapped ? " primary" : ""}`} disabled={busy}
                    onClick={() => act(() => setJunctionWiring(jid, !swapped))}
                    title="Use 'swapped' if this junction's east–west pins light its north/south heads">
                    {jid}: {swapped ? "EW ⇄ NS swapped" : "normal"}
                  </button>
                );
              })}
            </div>
            <span className="bd-hint">If a junction lights the wrong pair, swap it here instead of rewiring.</span>
          </div>
          <label className="bd-field">
            <span>Camera source</span>
            <div className="bd-row">
              <input className="bd-input" value={cameraInput ?? ""} onChange={(e) => setCameraInput(e.target.value)}
                placeholder="http://<phone-ip>:8080/video" spellCheck={false} />
              <button className="bd-btn primary" disabled={busy} onClick={() => act(() => setBoardCamera(cameraInput))}>Connect</button>
            </div>
          </label>
          <div className="bd-row">
            {CAMERA_PRESETS.map((p) => (
              <button key={p.value} className="bd-btn small" onClick={() => setCameraInput(p.value)}>{p.label}</button>
            ))}
          </div>
          <label className="bd-field">
            <span>Registered plates (optional)</span>
            <div className="bd-row">
              <input className="bd-input" value={platesInput ?? ""} onChange={(e) => setPlatesInput(e.target.value)}
                placeholder="TN07, KA01, MH12 …" spellCheck={false} />
              <button className="bd-btn primary" disabled={busy} onClick={() => act(() => setRegisteredPlates(platesInput ?? ""))}>Save</button>
            </div>
            <span className="bd-hint">The plates printed on your cars. A read one character off (KA07) snaps to the registered one (KA01).</span>
          </label>
          <label className="bd-field">
            <span>LED boards (USB)</span>
            <div className="bd-row">
              <select className="bd-select" value={portChoice ?? "auto"} onChange={(e) => setPortChoice(e.target.value)} onFocus={loadPorts}>
                <option value="auto">Auto-detect all Nanos</option>
                <option value="off">Off</option>
                {ports.map((p) => (
                  <option key={p.port} value={p.port}>{p.port} — {p.description}{p.arduino_like ? " (Arduino)" : ""}</option>
                ))}
              </select>
              <button className="bd-btn primary" disabled={busy} onClick={() => act(() => setSerialPort(portChoice ?? "auto"))}>Apply</button>
            </div>
          </label>
          {Object.entries(ser.boards || {}).map(([port, hello]) => (
            <div key={port} className="bd-hint"><i className="fas fa-check" /> {port}: {hello}</div>
          ))}
          <div className="bd-row">
            <button className="bd-btn danger" disabled={busy} onClick={() => act(resetBoardAI)}>
              <i className="fas fa-rotate-left" /> Reset AI
            </button>
            <span className="bd-hint">Tick {state.tick_s}s · min green {Math.round(state.tick_s * 10)}s</span>
          </div>
        </div>
      </div>
    </div>
  );
}

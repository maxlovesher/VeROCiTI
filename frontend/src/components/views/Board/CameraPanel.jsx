import { useEffect, useRef, useState } from "react";
import {
  setCamera,
  calibrateBoard,
  resetCalibration,
  captureEmptyBoard,
  frameUrl,
} from "../../../services/boardService";

const ROAD_LABEL = {
  road_VW1_J1: "J1 · west arm",
  road_J3_J1: "J1 · east link",
  road_VN1_J1: "J1 · north arm",
  road_VS1_J1: "J1 · south arm",
  road_J1_J3: "J3 · west link",
  road_J4_J3: "J3 · east arm",
  road_J2_J3: "J3 · north arm",
  road_J5_J3: "J3 · south arm",
};
const RUNG = {
  FULL_AI: "Full AI",
  HISTORICAL_PROFILE: "Time-of-day plan",
  LOCAL_ACTUATED: "Local queues only",
  FIXED_TIME: "Fixed time",
};
// Where each corner sits in the little board sketch (J1 box on the left, J3 box on the right).
const SPOTS = [[12, 12], [148, 12], [148, 58], [12, 58]];

function CornerGuide({ index }) {
  const [x, y] = SPOTS[index];
  return (
    <svg className="cp-guide" viewBox="0 0 160 70" role="img" aria-label="Which corner to click next">
      <rect x="12" y="12" width="30" height="46" rx="3" className="box" />
      <rect x="118" y="12" width="30" height="46" rx="3" className="box" />
      <rect x="42" y="27" width="76" height="16" className="road" />
      <text x="27" y="39" textAnchor="middle" className="t">J1</text>
      <text x="133" y="39" textAnchor="middle" className="t">J3</text>
      <circle cx={x} cy={y} r="7" className="spot" />
    </svg>
  );
}

export default function CameraPanel({ camera, mode, act, busy }) {
  const [source, setSource] = useState(null);
  const [view, setView] = useState("board");
  const [picking, setPicking] = useState(false);
  const [points, setPoints] = useState([]);              // fractions of the picture, 0..1
  const [frozen, setFrozen] = useState(0);
  const [tick, setTick] = useState(0);
  const imgRef = useRef(null);

  useEffect(() => {
    if (picking || !camera?.live) return undefined;      // hold the picture still while corners are placed on it
    const id = setInterval(() => setTick((t) => t + 1), 500);
    return () => clearInterval(id);
  }, [picking, camera?.live]);

  if (!camera) {
    return <section className="lb-card"><header className="lb-card-head"><h2>Camera</h2></header><p className="lb-hint">The camera module isn't loaded.</p></section>;
  }

  const calibrated = camera.calibrated;
  const shown = picking ? "raw" : calibrated ? view : "raw";
  const src = frameUrl(shown, picking ? frozen : tick);
  const sourceText = source ?? camera.source ?? "";

  const toFraction = (e) => {
    const r = imgRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    const clamp = (v) => Math.min(1, Math.max(0, v));
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)];
  };

  const startPicking = () => {
    setPoints([]);
    setFrozen(Date.now());
    setPicking(true);
  };

  const addPoint = (e) => {
    if (!picking || points.length >= 4) return;
    const p = toFraction(e);
    if (p) setPoints([...points, p]);
  };

  const dragPoint = (e, i) => {
    e.preventDefault();
    e.stopPropagation();
    const move = (ev) => {
      const p = toFraction(ev);
      if (p) setPoints((cur) => cur.map((q, k) => (k === i ? p : q)));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const save = () => act(async () => {
    await calibrateBoard(points);
    const state = await captureEmptyBoard();            // new corners drop the old empty-board photo; the board is empty now
    setPicking(false);
    setPoints([]);
    return state;
  });

  const ambulances = (camera.vehicles || []).filter((v) => v.kind === "ambulance");
  const cars = (camera.vehicles || []).filter((v) => v.kind === "car");
  const plateText = (v) => {
    if (v.count > 1) return `${v.count} cars`;               // nose to tail: one blob, several plates
    if (v.plate) return v.plate;
    return camera.plates_available === false ? `Car ${v.id}` : "reading…";
  };
  const followNote = camera.realigned_s_ago != null && camera.realigned_s_ago < 90
    ? `Re-aligned to the moved phone ${camera.realigned_s_ago} s ago`
    : camera.following ? "Following the phone if it moves" : "";

  return (
    <section className="lb-card cp">
      <header className="lb-card-head">
        <h2>Camera</h2>
        <span className={`cp-pill ${camera.live ? "ok" : "bad"}`}>
          {camera.live ? `Live · ${camera.fps || "…"} fps` : camera.source ? (camera.error || "No picture") : "Not connected"}
        </span>
      </header>

      <form className="cp-source" onSubmit={(e) => { e.preventDefault(); act(() => setCamera(sourceText.trim())); }}>
        <input value={sourceText} onChange={(e) => setSource(e.target.value)} spellCheck={false}
          placeholder="Phone stream, e.g. http://172.20.10.2:8080/video" aria-label="Camera stream address" />
        <button type="submit" className="lb-btn primary" disabled={busy}>{camera.source ? "Reconnect" : "Connect"}</button>
      </form>

      <div className="cp-body">
        <div>
          <div className={`cp-feed${picking ? " picking" : ""}`} onClick={addPoint}>
            {camera.live ? (
              <div className="cp-frame">
                <img ref={imgRef} src={src} alt={shown === "raw" ? "Camera picture" : "Flat plan of the board"} draggable={false} />
                {picking && points.map((p, i) => (
                  <button key={i} type="button" className="cp-dot" style={{ left: `${p[0] * 100}%`, top: `${p[1] * 100}%` }}
                    onPointerDown={(e) => dragPoint(e, i)} onClick={(e) => e.stopPropagation()} aria-label={`Corner ${i + 1}`}>{i + 1}</button>
                ))}
              </div>
            ) : (
              <div className="cp-empty">{camera.source ? "Waiting for the camera…" : "Connect the phone to see the board."}</div>
            )}
          </div>

          <div className="cp-actions">
            {!picking ? (
              <>
                <button className="lb-btn primary" disabled={!camera.live || busy} onClick={startPicking}>
                  {calibrated ? "Recalibrate" : "Calibrate"}
                </button>
                {calibrated && (
                  <div className="lb-seg" role="group" aria-label="Picture shown">
                    <button className={view === "board" ? "active" : ""} onClick={() => setView("board")}>Board plan</button>
                    <button className={view === "raw" ? "active" : ""} onClick={() => setView("raw")}>Camera</button>
                  </div>
                )}
                <button className="lb-btn" disabled={!calibrated || !camera.live || busy} onClick={() => act(captureEmptyBoard)}
                  title="Photograph the board with no cars on it, so lane paint and glare are ignored">
                  Photograph empty board
                </button>
                {calibrated && <button className="lb-btn" disabled={busy} onClick={() => act(resetCalibration)}>Reset</button>}
              </>
            ) : (
              <>
                <button className="lb-btn primary" disabled={points.length < 4 || busy} onClick={save}>Save corners</button>
                <button className="lb-btn" disabled={!points.length || busy} onClick={() => setPoints(points.slice(0, -1))}>Undo</button>
                <button className="lb-btn" disabled={busy} onClick={() => { setPicking(false); setPoints([]); }}>Cancel</button>
              </>
            )}
          </div>
        </div>

        <aside className="cp-side">
          {picking ? (
            <>
              <h3>{points.length < 4 ? `Click corner ${points.length + 1} of 4` : "Drag any dot to fine-tune, then save"}</h3>
              <p className="cp-corner">{camera.labels?.[Math.min(points.length, 3)]}</p>
              <CornerGuide index={Math.min(points.length, 3)} />
              <p className="lb-hint">Click the outer corners of the two black junction boxes, clockwise. North is toward J2 (the top of the board), however the phone is turned. Keep cars off the board: it photographs the empty board when you save.</p>
            </>
          ) : (
            <>
              <h3>{calibrated ? (camera.has_reference ? "Calibrated · empty board saved" : "Calibrated · photograph the empty board") : "Not calibrated"}</h3>
              {mode === "camera" && camera.live && calibrated && (
                <p className="cp-rung">AI rung: <strong>{RUNG[camera.control_mode] || camera.control_mode}</strong></p>
              )}
              {followNote && <p className="cp-follow">{followNote}</p>}
              {ambulances.length > 0 && <div className="cp-amb">Ambulance on {ROAD_LABEL[ambulances[0].road] || "the board"}</div>}
              {cars.length === 0 && ambulances.length === 0 ? (
                <p className="lb-hint">{calibrated ? "No cars on the board." : "Calibrate to start counting cars."}</p>
              ) : (
                <ul className="cp-list">
                  {cars.map((v) => (
                    <li key={v.id}>
                      <span className="cp-plate">
                        <strong>{plateText(v)}</strong>
                        {v.plate_state && <em>{v.plate_state}</em>}
                      </span>
                      <span>{ROAD_LABEL[v.road] || "in the junction"}</span>
                    </li>
                  ))}
                </ul>
              )}
              {mode !== "camera" && calibrated && <p className="lb-hint">Choose <strong>AI from camera</strong> above to let these cars drive the lights.</p>}
            </>
          )}
        </aside>
      </div>
    </section>
  );
}

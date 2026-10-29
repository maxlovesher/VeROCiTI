"""
led_board.py — the Signal AI's lights on the real board.
=========================================================
One Arduino Uno switches the board's LEDs. It never decides anything: the laptop sends
one line per tick over USB and the Uno shows it (and flashes amber if the laptop goes quiet).

    S<J1 EW><J1 NS><J2 EW><J2 NS>...<J5 EW><J5 NS>\\n      e.g. SGRRRRGRRRR

Each character is R / Y / G (that head pair's lamp), O (off) or W (all three lamps on).
A head pair is the two heads that face each other along a road (west+east = EW,
north+south = NS), so one character drives two LEDs of the same colour.

Modes
  follow  the board mirrors the Signal AI running in the simulation (the default)
  camera  the Signal AI drives the board from the cars the overhead camera sees (board_camera.py)
  test    every head in turn, one lamp at a time (finds a mis-wired lamp)
  allon   every lamp of every head on at once (finds a dead LED)
  wiring  J1 west+east green and J3 north+south green, everything else red: the two
          junctions are in OPPOSITE states, so a crossed pair cannot be mistaken for a fine one
  manual  each head pair set by hand

Wiring: on the built board a junction's two pairs can be connected the other way round.
`swapped` fixes that in one place (EW and NS trade places on the wire) so the AI, the
dashboard and manual control all keep their normal meaning. `exchanged` does the same when
the two junctions are on each other's pins (J1 on the camera's left is wired to J3's slot).
"""

import atexit
import json
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

JUNCTIONS = ("J1", "J2", "J3", "J4", "J5")        # frame order; the Uno sketch reads its own slice
BUILT = ("J1", "J3")                              # the junctions that exist on the real board
DIRECTIONS = ("EW", "NS")
LAMPS = "RYGOW"
MODES = ("follow", "camera", "test", "allon", "wiring", "manual")
BAUD = 115200
TICK_S = 0.4
RESEND_S = 0.25                                   # the sketch's fail-safe is 3 s
OPEN_SETTLE_S = 2.0                               # opening the port resets the Uno; let it boot
RETRY_S = 2.0
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403}   # Arduino, Arduino.org, CH340, FTDI
CONFIG_PATH = os.environ.get("LED_BOARD_CONFIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "led_board_config.json"))
# How the demo board is wired (see hardware/README.md): no head pair is crossed, but the junction on the camera's
# left sits on the J3 pins and the right one on J1's. Anyone who builds it the other way flips these on the LED Board page.
DEFAULT_SWAPPED = ()
DEFAULT_EXCHANGED = True

Signals = Dict[str, Dict[str, str]]


def all_signals(state: str) -> Signals:
    return {j: {d: state for d in DIRECTIONS} for j in JUNCTIONS}


def agent_signals(agent) -> Dict[str, str]:
    """R/Y/G for each phase group of one junction, from the Signal AI agent's own state."""
    out = {p: "R" for p in agent.phase_names}
    if agent.is_all_red:
        return out
    out[agent.phase_names[agent.current_phase]] = "Y" if agent.is_yellow else "G"
    return out


EXCHANGE = {"J1": "J3", "J3": "J1"}               # which wire slot each built junction uses when the two are exchanged


def to_wire(signals: Signals, swapped, exchanged: bool = False) -> Signals:
    """
    What goes on the wire: a swapped junction sends its EW state as NS and the other way round;
    with the junctions exchanged, J1 uses J3's pins and J3 uses J1's.
    """
    out: Signals = {j: {"EW": "O", "NS": "O"} for j in JUNCTIONS}
    for j in JUNCTIONS:
        st = signals.get(j, {})
        ew, ns = st.get("EW", "O"), st.get("NS", "O")
        slot = EXCHANGE.get(j, j) if exchanged else j
        out[slot] = {"EW": ns, "NS": ew} if j in swapped else {"EW": ew, "NS": ns}
    return out


def encode_frame(signals: Signals) -> str:
    chars = []
    for j in JUNCTIONS:
        st = signals.get(j, {})
        for d in DIRECTIONS:
            c = st.get(d, "O")
            chars.append(c if c in LAMPS else "O")
    return "S" + "".join(chars) + "\n"


def lamp_test(now: float) -> Signals:
    """Every built head pair in turn, one lamp at a time: red, yellow, green."""
    steps = [(j, d, c) for j in BUILT for d in DIRECTIONS for c in "RYG"]
    j, d, c = steps[int(now / 0.7) % len(steps)]
    out = all_signals("O")
    out[j][d] = c
    return out


def wiring_check() -> Signals:
    """J1 west+east and J3 north+south green, the rest red: opposite states on the two junctions."""
    out = all_signals("O")
    for j in BUILT:
        out[j] = {"EW": "R", "NS": "R"}
    out["J1"]["EW"] = "G"
    out["J3"]["NS"] = "G"
    return out


WIRING_CHECK_EXPECT = ("Look at the board: only J1's west and east heads and J3's north and south heads should be green; "
                       "the other four head pairs red. If the two junctions are the wrong way round (left shows J3's pattern), "
                       "flip 'Junctions on each other's pins'; if the pairs inside a junction are the wrong way round, flip "
                       "'Pairs wired crossed' on that junction.")


def _looks_like_arduino(p) -> bool:
    desc = f"{p.description or ''} {p.manufacturer or ''}".lower()
    return (p.vid in ARDUINO_VIDS) or "arduino" in desc or "ch340" in desc or "usb-serial" in desc


class UnoLink:
    """The USB connection to the Uno: finds it, keeps the latest frame flowing, reconnects, and lets go cleanly."""

    def __init__(self, setting: str = "auto", start: bool = True):
        self.setting = setting                    # "auto" | "off" | a port name
        self.port: Optional[str] = None
        self.error: Optional[str] = None
        self.frames_sent = 0
        self.last_frame: Optional[str] = None
        self._ser = None
        self._frame: Optional[str] = None
        self._opened_at = 0.0
        self._last_try = 0.0
        self._last_write = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        atexit.register(self.close)
        if start:
            self._thread.start()

    @property
    def connected(self) -> bool:
        return self._ser is not None

    def send(self, frame: str) -> None:
        self._frame = frame

    def set_port(self, setting: str) -> None:
        self.close()
        self.setting = setting or "auto"
        self.error = None

    def close(self) -> None:
        with self._lock:
            ser, self._ser, self.port = self._ser, None, None
        if ser is not None:
            try:
                ser.close()                       # a port left open can wedge Windows until the USB is replugged
            except Exception:
                pass

    @staticmethod
    def list_ports() -> List[Dict[str, Any]]:
        try:
            from serial.tools import list_ports
        except Exception:
            return []
        return [{"port": p.device, "description": p.description or "", "arduino": _looks_like_arduino(p)}
                for p in list_ports.comports()]

    def _pick_port(self) -> Optional[str]:
        if self.setting not in ("auto", ""):
            return self.setting
        for p in self.list_ports():
            if p["arduino"]:
                return p["port"]
        return None

    def _open(self) -> None:
        try:
            import serial
        except Exception:
            self.error = "pyserial isn't installed (pip install pyserial)"
            return
        port = self._pick_port()
        if not port:
            self.error = "No Arduino found: use a USB data cable and check the port list"
            return
        try:
            ser = serial.Serial(port, BAUD, timeout=0.2, write_timeout=1.0)
        except Exception as e:
            self.error = f"Couldn't open {port}: {e}"
            return
        with self._lock:
            self._ser, self.port, self.error = ser, port, None
        self._opened_at = time.time()

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            try:
                if self.setting == "off":
                    self.close()
                elif self._ser is None:
                    if now - self._last_try >= RETRY_S:
                        self._last_try = now
                        self._open()
                elif now - self._opened_at >= OPEN_SETTLE_S and self._frame and now - self._last_write >= RESEND_S:
                    self._last_write = now
                    with self._lock:
                        if self._ser is not None:
                            self._ser.write(self._frame.encode())
                    self.frames_sent += 1
                    self.last_frame = self._frame.strip()
            except Exception as e:
                self.error = f"Lost the Uno: {e}"
                self.close()
            time.sleep(0.05)


class LedBoard:
    def __init__(self, sim_signals: Optional[Callable[[], Signals]] = None, link: Optional[UnoLink] = None,
                 config_path: str = CONFIG_PATH, start: bool = True):
        self.sim_signals = sim_signals
        self.config_path = config_path
        self.lock = threading.RLock()
        cfg = self._load()
        self.swapped = {j for j in cfg.get("swapped", DEFAULT_SWAPPED) if j in JUNCTIONS}
        self.exchanged = bool(cfg.get("exchanged", DEFAULT_EXCHANGED))
        self.link = link or UnoLink(setting=cfg.get("port", "auto"), start=start)
        self.pilot = None                                   # board_camera.CameraPilot, when the camera module is available
        self.mode = "follow"
        self.manual: Signals = all_signals("R")
        self.signals: Signals = all_signals("O")          # what each head pair should show (logical, before wiring)
        self.sim_error: Optional[str] = None
        if start:
            threading.Thread(target=self._loop, daemon=True).start()

    # ── settings ──
    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        cfg = self._load()                                  # the camera keeps its own keys in this file: only touch ours
        cfg.update({"swapped": sorted(self.swapped), "exchanged": self.exchanged, "port": self.link.setting})
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except OSError:
            pass

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        with self.lock:
            if mode == "manual" and self.mode != "manual":
                self.manual = {j: dict(self.signals.get(j, {"EW": "R", "NS": "R"})) for j in JUNCTIONS}   # start from what's showing
            self.mode = mode

    def set_manual(self, junction: str, direction: str, lamp: str) -> None:
        lamp = str(lamp or "").upper()
        if lamp not in LAMPS:
            raise ValueError("lamp must be R, Y, G, O or W")
        junctions = list(JUNCTIONS) if junction == "*" else [junction]
        directions = list(DIRECTIONS) if direction == "*" else [direction]
        if any(j not in JUNCTIONS for j in junctions):
            raise ValueError(f"junction must be one of {', '.join(JUNCTIONS)} or *")
        if any(d not in DIRECTIONS for d in directions):
            raise ValueError("direction must be EW, NS or *")
        with self.lock:
            self.set_mode("manual")
            for j in junctions:
                for d in directions:
                    self.manual[j][d] = lamp

    def set_swapped(self, junction: str, swapped: bool) -> List[str]:
        if junction not in JUNCTIONS:
            raise ValueError(f"junction must be one of {', '.join(JUNCTIONS)}")
        with self.lock:
            (self.swapped.add if swapped else self.swapped.discard)(junction)
            self._save()
            return sorted(self.swapped)

    def set_exchanged(self, exchanged: bool) -> bool:
        with self.lock:
            self.exchanged = bool(exchanged)
            self._save()
            return self.exchanged

    def set_port(self, setting: str) -> None:
        self.link.set_port(str(setting or "auto"))
        self._save()

    # ── the control loop ──
    def tick(self, now: Optional[float] = None) -> Signals:
        now = time.time() if now is None else now
        with self.lock:
            if self.mode == "test":
                signals = lamp_test(now)
            elif self.mode == "allon":
                signals = all_signals("W")
            elif self.mode == "wiring":
                signals = wiring_check()
            elif self.mode == "manual":
                signals = {j: dict(v) for j, v in self.manual.items()}
            else:
                try:
                    signals = self._from_camera(now) if self.mode == "camera" else self._from_sim()
                    self.sim_error = None
                except Exception as e:
                    self.sim_error = str(e)
                    return self.signals                         # send nothing new: the Uno flashes amber if this lasts
            self.signals = signals
        self.link.send(encode_frame(to_wire(signals, self.swapped, self.exchanged)))
        return signals

    def _from_sim(self) -> Signals:
        if not self.sim_signals:
            raise RuntimeError("the Signal AI simulation isn't connected")
        sim = self.sim_signals()
        out = all_signals("O")
        for j in BUILT:
            st = sim.get(j)
            if st is None:
                raise RuntimeError(f"the simulation has no {j}")
            out[j] = {d: st.get(d, "R") for d in DIRECTIONS}
        return out

    def _from_camera(self, now: float) -> Signals:
        if self.pilot is None:
            raise RuntimeError("the camera module isn't available")
        self.pilot.tick(now)                                    # cars in, the AI decides (at most one step per 0.8 s)
        agents = self.pilot.coordinator.agents
        out = all_signals("O")
        for j in BUILT:
            out[j] = agent_signals(agents[j])
        return out

    def _loop(self) -> None:
        while True:
            started = time.time()
            try:
                self.tick(started)
            except Exception as e:
                self.sim_error = str(e)
            time.sleep(max(0.0, TICK_S - (time.time() - started)))

    def state(self) -> Dict[str, Any]:
        with self.lock:
            signals = {j: dict(self.signals.get(j, {})) for j in BUILT}
            return {
                "ok": True,
                "mode": self.mode,
                "modes": list(MODES),
                "junctions": list(BUILT),
                "signals": signals,
                "manual": {j: dict(self.manual[j]) for j in BUILT},
                "swapped": sorted(self.swapped),
                "exchanged": self.exchanged,
                "frame": encode_frame(to_wire(self.signals, self.swapped, self.exchanged)).strip(),
                "link": {"connected": self.link.connected, "port": self.link.port, "setting": self.link.setting,
                         "error": self.link.error, "frames_sent": self.link.frames_sent},
                "sim_error": self.sim_error,
                "wiring_hint": WIRING_CHECK_EXPECT if self.mode == "wiring" else None,
                "camera": self.pilot.state() if self.pilot is not None else None,
            }


def register_led_board(app, sim_signals: Callable[[], Signals]) -> LedBoard:
    from flask import jsonify, request

    board = LedBoard(sim_signals=sim_signals)
    try:                                                        # the camera is optional: the LEDs work without it
        from board_camera import CameraPilot, register_camera_routes
        board.pilot = CameraPilot()
        register_camera_routes(app, board.pilot, board)
    except Exception as e:
        print(f"[LedBoard] Note: camera module not loaded: {e}")

    def bad(msg: str, code: int = 400):
        return jsonify({"ok": False, "error": msg}), code

    def reply():
        return jsonify(board.state())

    @app.route("/api/board/state")
    def led_state():
        return reply()

    @app.route("/api/board/ports")
    def led_ports():
        return jsonify({"ok": True, "ports": board.link.list_ports()})

    @app.route("/api/board/mode", methods=["POST"])
    def led_mode():
        try:
            board.set_mode(str((request.get_json(silent=True) or {}).get("mode", "")))
        except ValueError as e:
            return bad(str(e))
        return reply()

    @app.route("/api/board/manual", methods=["POST"])
    def led_manual():
        d = request.get_json(silent=True) or {}
        try:
            board.set_manual(str(d.get("junction", "")), str(d.get("direction", "")), str(d.get("state", "")))
        except ValueError as e:
            return bad(str(e))
        return reply()

    @app.route("/api/board/wiring", methods=["POST"])
    def led_wiring():
        d = request.get_json(silent=True) or {}
        try:
            if "exchanged" in d:
                board.set_exchanged(bool(d["exchanged"]))
            if "junction" in d:
                board.set_swapped(str(d.get("junction", "")), bool(d.get("swapped", True)))
        except ValueError as e:
            return bad(str(e))
        return reply()

    @app.route("/api/board/serial", methods=["POST"])
    def led_serial():
        board.set_port(str((request.get_json(silent=True) or {}).get("port", "auto")))
        return reply()

    return board

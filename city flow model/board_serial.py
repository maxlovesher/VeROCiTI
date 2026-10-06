"""
board_serial.py — USB serial link to the Arduino(s) driving the board's LEDs.
=============================================================================
The lamps show exactly what the Signal AI decided: every control tick the
controller's signal state is encoded as one line and written to every
connected board. The boards make no decisions of their own.

Protocol (115200 baud, one line per update):

    S<J1 EW><J1 NS><J2 EW><J2 NS>...<J5 EW><J5 NS>\n     e.g. SGRRGYRGRRG\n

with each character one of R / Y / G (that head's lamp) or O (all off,
used by the lamp test). Two Nanos share the job — Nano A drives J1–J3 and
Nano B drives J4–J5 — and both simply receive the same line and pick out
their own junctions. A board that hears nothing for 3 seconds flashes amber
on all its heads (the usual controller fail-safe), so a pulled USB cable
mid-demo is safe.
"""

import threading
import time
from typing import Dict, List, Optional

import board_layout as bl

BAUD = 115200
# USB vendor ids of the usual Arduino boards and clones (Arduino, CH340, CP210x, FTDI).
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x10C4, 0x0403}
VALID = set("RYGO")
RETRY_S = 2.0


def encode_frame(signals: Dict[str, Dict[str, str]]) -> str:
    chars = []
    for jid in bl.SIGNALLED:
        st = signals.get(jid, {})
        for phase in ("EW", "NS"):
            c = st.get(phase, "R")
            chars.append(c if c in VALID else "R")
    return "S" + "".join(chars) + "\n"


def list_ports() -> List[Dict[str, str]]:
    try:
        from serial.tools import list_ports as lp
    except Exception:
        return []
    return [
        {
            "port": p.device,
            "description": p.description or "",
            "arduino_like": p.vid in ARDUINO_VIDS if p.vid is not None else False,
        }
        for p in lp.comports()
    ]


def _wanted_ports(setting: str) -> List[str]:
    """'auto' = every Arduino-looking USB port, 'off' = none, else a comma-separated list like 'COM5,COM7'."""
    if setting == "off":
        return []
    if setting == "auto":
        return [p["port"] for p in list_ports() if p["arduino_like"]]
    return [p.strip() for p in setting.split(",") if p.strip()]


class BoardSerial:
    """Keeps every wanted board connected in the background, and sends each one the newest frame."""

    def __init__(self, port: str = "auto"):
        self.port_setting = port
        self._conns: Dict[str, object] = {}
        self._lock = threading.Lock()
        self._frame: Optional[str] = None
        self._wake = threading.Event()
        self.errors: Dict[str, str] = {}
        self.boards: Dict[str, str] = {}
        self.frames_sent = 0
        self.last_sent: Optional[str] = None
        threading.Thread(target=self._run, daemon=True).start()

    def set_port(self, port: str) -> None:
        with self._lock:
            self.port_setting = (port or "auto").strip()
            for p in list(self._conns):
                self._close(p)
            self.errors = {}
            self.boards = {}
        self._wake.set()

    def send(self, frame: str) -> None:
        self._frame = frame
        self._wake.set()

    def status(self) -> Dict:
        ports = sorted(self._conns)
        return {
            "setting": self.port_setting,
            "connected": bool(ports),
            "ports": ports,
            "boards": dict(self.boards),
            "errors": dict(self.errors),
            "frames_sent": self.frames_sent,
            # The AI's latest output line, shown even when no board is plugged in.
            "last_frame": (self._frame or "").strip() or None,
        }

    def _close(self, port: str) -> None:
        ser = self._conns.pop(port, None)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        self.boards.pop(port, None)

    def _connect_missing(self) -> None:
        try:
            import serial
        except Exception:
            self.errors = {"pyserial": "pyserial not installed (pip install pyserial)"}
            return
        wanted = _wanted_ports(self.port_setting)
        if not wanted:
            self.errors = {} if self.port_setting == "off" else {"usb": "No Arduino found on USB"}
            return
        self.errors.pop("usb", None)
        opened = []
        for port in wanted:
            if port in self._conns:
                continue
            try:
                self._conns[port] = serial.Serial(port, BAUD, timeout=0, write_timeout=0.5)
                self.errors.pop(port, None)
                opened.append(port)
            except Exception as e:
                self.errors[port] = f"Could not open {port}: {e}"
        if opened:
            # Opening a port resets a Nano/Uno; give the bootloader time before talking.
            time.sleep(2.0)

    def _read_hello(self, port: str, ser) -> None:
        try:
            data = ser.read(256)
        except Exception:
            return
        for line in data.decode(errors="ignore").splitlines():
            if line.startswith("VEROCITI"):
                self.boards[port] = line.strip()

    def _run(self) -> None:
        last_attempt = 0.0
        while True:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            with self._lock:
                if self.port_setting != "off" and time.time() - last_attempt >= RETRY_S:
                    last_attempt = time.time()
                    self._connect_missing()
                frame = self._frame
                if not frame or not self._conns:
                    continue
                for port, ser in list(self._conns.items()):
                    try:
                        ser.write(frame.encode("ascii"))
                        self._read_hello(port, ser)
                    except Exception as e:
                        self.errors[port] = f"Serial write failed: {e}"
                        self._close(port)
                self.frames_sent += 1
                self.last_sent = frame

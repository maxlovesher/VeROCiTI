// LED board: the Signal AI's lights on the real board (server: city flow model/led_board.py).
const API = import.meta.env.VITE_CITYFLOW_API || '/api';

async function call(path, body) {
  const res = await fetch(`${API}/board${path}`, body === undefined ? {} : {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.ok === false) throw new Error(data.error || `Board API error: ${res.status}`);
  return data;
}

export const fetchBoardState = () => call('/state');
export const listPorts = () => call('/ports');
// mode: 'follow' | 'test' | 'allon' | 'wiring' | 'manual'
export const setBoardMode = (mode) => call('/mode', { mode });
// junction 'J1' | 'J3' | '*', direction 'EW' | 'NS' | '*', lamp 'R' | 'Y' | 'G' | 'O' (off) | 'W' (all three on)
export const setManualLamp = (junction, direction, state) => call('/manual', { junction, direction, state });
// swapped: this junction's two head pairs are connected the other way round
export const setWiring = (junction, swapped) => call('/wiring', { junction, swapped });
// exchanged: the two junctions are on each other's pins (the one on the camera's left uses J3's)
export const setExchanged = (exchanged) => call('/wiring', { exchanged });
// 'auto' | 'off' (lets go of the port) | a port name such as 'COM9'
export const setPort = (port) => call('/serial', { port });

// Camera: the phone's stream URL (e.g. http://172.20.10.2:8080/video), or '' to disconnect.
export const setCamera = (source) => call('/camera', { source });
// The four corners of the black boxes, clockwise from J1's north-west, as fractions (0..1) of the picture.
export const calibrateBoard = (points) => call('/calibrate', { points });
export const resetCalibration = () => call('/calibrate', { reset: true });
export const captureEmptyBoard = () => call('/reference', {});
export const clearEmptyBoard = () => call('/reference', { clear: true });
// view: 'raw' (what the phone sees) | 'board' (the flat plan with cars drawn on); t busts the cache.
export const frameUrl = (view, t) => `${API}/board/frame.jpg?view=${view}&t=${t}`;

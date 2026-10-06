// Physical demo board: overhead camera -> Signal AI -> LEDs (server: city flow model/board_api.py).
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
export const listSerialPorts = () => call('/serial/ports');

export const setBoardCamera = (source) => call('/camera', { source });
export const calibrateBoard = (corners) => call('/calibrate', { corners });
// One click in the middle of each junction box; the backend finds the exact edges itself.
export const autoCalibrateBoard = (centres) => call('/autocalibrate', { centres });
export const resetCalibration = () => call('/calibrate', { reset: true });
export const captureEmptyBoard = () => call('/reference', {});
export const clearEmptyBoard = () => call('/reference', { clear: true });
export const setBoardMode = (mode) => call('/mode', { mode });
export const setBoardLayout = (layout) => call('/layout', { layout });
export const setRegisteredPlates = (plates) => call('/plates', { plates });
export const setJunctionWiring = (junction, swapped) => call('/wiring', { junction, swapped });
export const setSerialPort = (port) => call('/serial', { port });
export const resetBoardAI = () => call('/reset', {});
// Manual control: junction 'J1'..'J5' or '*', direction 'EW' | 'NS' | '*', state 'R' | 'Y' | 'G' | 'O'.
export const setManualLight = (junction, direction, state) => call('/manual', { junction, direction, state });
export const revertManualControl = () => call('/manual/revert', {});
export const toggleSyntheticCar = (u, v, kind = 'car') => call('/synthetic/toggle', { u, v, kind });
export const clearSyntheticCars = () => call('/synthetic/clear', {});

export const boardFrameUrl = (view, t) => `${API}/board/frame.jpg?view=${view}&t=${t}`;

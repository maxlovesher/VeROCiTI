# VeROCiTI LED board

A table-top model of two junctions (J1 and J3) whose lights are the **Signal AI's** lights. One Arduino Uno
switches 24 LEDs; a phone overhead can watch model cars so the AI decides from what is really on the board.

```
phone camera ──WiFi──► laptop ──► board_camera.py (cars, ambulance) ─┐
                                                                      ▼
Signal AI (agent.py, unchanged) ───────────────────────► led_board.py ──USB──► Arduino Uno ──► 24 LEDs
```

The Uno never decides anything. If the laptop goes quiet for 3 s it flashes amber, like a real controller
that has lost its brain.

## Run it (fresh clone, new laptop)

1. Install **Python 3.10+** (tick *Add to PATH*) and **Node.js 18+**. For a new Uno also install the
   [Arduino IDE](https://www.arduino.cc/en/software) once (it carries the uploader).
2. `git clone https://github.com/maxlovesher/VeROCiTI.git`
3. **New Uno only:** plug it in with a USB *data* cable and run **`hardware\upload_sketch.bat`**. It finds the Uno,
   installs the AVR board package the first time, and uploads `verociti_board.ino`. Add the port if it can't tell
   (`upload_sketch.bat COM5`); `upload_sketch.bat check` only compiles. An Uno that already has the sketch skips this.
4. Wire the LEDs as in the table below, then double-click **`start.bat`** in the repo folder. The first run
   installs the Python and dashboard packages (several minutes) and opens the dashboard.
5. Open **LED Board**. The top left says *Arduino Uno on COMx* by itself, with no port to pick.
6. Check the lights in this order: **Lamp test** (one lamp at a time), **All on** (all 24 LEDs, keep it short),
   **Wiring check** (J1 west+east and J3 north+south green, the rest red; on the camera picture J1 is the left box).
   The defaults match the demo board. If the wrong *pairs* light, flip **Pairs wired crossed** on that junction;
   if the left junction shows what the right should, flip **Left and right junction are on each other's pins**
   under *Connection*.
7. **Follow AI** mirrors the Signal AI running in the simulation. No camera needed.

On macOS or Linux run the same three things by hand: `pip install -r "city flow model/requirements.txt"`,
`python "city flow model/server_standalone.py"` and, in `frontend/`, `npm install && npm run dev`
(then open http://localhost:5173). Upload the sketch from the Arduino IDE.

## Wiring

The two heads that face each other along a road share one pin per colour (join their R wires, their Y wires and
their G wires, then one jumper to the pin). All LED GNDs go to the Uno's GND.

| Junction | Pair | Red | Yellow | Green |
|---|---|---|---|---|
| J1 | east–west (west + east heads) | D2 | D3 | D4 |
| J1 | north–south (north + south heads) | D5 | D6 | D7 |
| J3 | east–west | A0 | A1 | A2 |
| J3 | north–south | A3 | A4 | A5 |

On the demo board the junction on the **left** of the camera picture (software J1) is wired to the **A0–A5** group and the
one on the right to **D2–D7**; the software's *Left and right junction are on each other's pins* switch (on by default)
accounts for that. Wire yours the other way and just turn the switch off.

Don't use D0/D1 (USB). Lamps lit when they should be off? Your modules share VCC: set `#define ACTIVE_HIGH 0` and re-upload.
Uploading: Arduino IDE → Board **Arduino Uno** → upload; free the port first by setting *USB port* to **Off** on the page.

The sketch reads one line per update: `S` + one letter per head pair for J1 EW, J1 NS, J2 EW, J2 NS, J3 EW, J3 NS, …
`R` red, `Y` amber, `G` green, `O` off, `W` all three lamps on.

## Using the camera

1. Phone and laptop on the same hotspot, start **IP Webcam**, and under *Camera* enter `http://<phone-ip>:8080/video` → **Connect**.
2. With no cars on the board press **Calibrate** and click the **outer corners of the two black junction boxes**,
   clockwise from J1's north-west corner (the sketch beside the picture shows which one is next). Drag a dot to
   fine-tune, then **Save corners**. It photographs the empty board straight away.
3. Choose **AI from camera**. White cards are cars; a **pale pink card reading AMBULANCE** opens a green corridor
   (a red box is drawn round it in the *Board plan* view). If the camera goes quiet the AI steps down to a fixed-time plan.

If the phone is nudged the page follows it by itself; if it is moved a lot, calibrate again. The phone's address,
the corners and the empty-board photo belong to one laptop and one table, so they are saved next to the code in
`city flow model/led_board_config.json` and two `.png` files that git ignores: every clone starts fresh and just
needs its own connect-and-calibrate (about a minute). The address is remembered across restarts, but a phone
hotspot hands out a new one now and then: if the picture is dead, read the address off the IP Webcam screen and press **Reconnect**.

## If something goes wrong

| Symptom | Fix |
|---|---|
| *Not connected* | Use a USB **data** cable; pick the port by hand under *Connection*. |
| All lamps flash amber | The Uno isn't hearing the laptop: check the status line, or reconnect. |
| The port won't open after a crash | Set *USB port* to **Off**, then replug the Uno's USB cable. |
| Cars missing or ghost cars | Calibrate again, then *Photograph empty board* with nothing on it. Kill glare. |
| Lights look reversed | **Pairs wired crossed** on that junction. |

Tests (no hardware needed): `python "city flow model/scripts/test_led_board.py"` and `python "city flow model/scripts/test_board_camera.py"`.

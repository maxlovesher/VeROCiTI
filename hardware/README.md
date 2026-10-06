# VeROCiTI physical demo board

A table-top model of the 5-junction network the Signal AI already runs. A phone overhead watches
model cars, the **Signal AI decides every light**, and two Arduino Nanos switch the LEDs to match.

```
phone camera ──WiFi──► laptop: board_vision (finds cars, reads plates)
                               ► Signal AI (agent.MultiAgentCoordinator, unchanged)
                               ► "SGRRGYRGRRG" over USB ──► Nano A (J1–J3) + Nano B (J4–J5) ──► LEDs
```

The Nanos never decide anything. If the laptop goes quiet for 3 s they flash amber, like a real
controller that has lost its brain.

## Presenting the board from another laptop

The repository ships the settings for the built board (two junctions J1 + J3, J3's head pairs wired
the other way round), so a fresh laptop starts right. Only the camera, calibration and empty-board
shot are per-laptop:

1. **Install** Python 3.10+ and Node.js 18+, clone or download this repo, then double-click **`start.bat`**.
   The first run installs everything (a few minutes) and opens the dashboard on **Physical Board**.
2. **Plug in the Arduino Uno** (sketch `hardware/verociti_board/verociti_board.ino`, `BOARD 'A'`, already on it).
   The *LED boards* status should show its COM port. Click **Lamp test** to check every head lights.
3. **Phone camera:** connect the phone and laptop to the same hotspot, start **IP Webcam**, and enter
   `http://<phone-ip>:8080/video` under *Camera source* → **Connect**.
4. **Calibrate:** with the cars off the board, click **Recalibrate**, then the middle of **J1's** junction box,
   then the middle of **J3's**. Wait about 8 s (the lights go dark while it measures; it also captures the
   empty board). After that it follows the phone by itself if it gets nudged.
5. Put the cars back and switch to **AI from camera**. The pink **AMBULANCE** card opens a green corridor.

If J3 lights the wrong pair, use *Setup → Head wiring* to toggle it.

## Quick build: one junction (the default)

The dashboard's **Board layout** defaults to *One junction (J1)*. Everything below still applies, but smaller:

- **Parts:** 1 Nano + USB data cable, 4 LED modules (+1 spare), 1 breadboard, ~16 female-to-male
  and ~7 male-to-male jumpers, a 50 × 50 cm board, and a phone mount.
- **Board:** two 8 cm matte-black roads crossing in the middle, running edge to edge (50 cm tip to tip).
- **Heads:** one at each corner of the crossing, 1–2 cm off the road.
  NW + SE heads = **EW** → D2 R, D3 Y, D4 G. NE + SW heads = **NS** → D5 R, D6 Y, D7 G.
  Each pair's matching wires share one breadboard row. All 4 GNDs go to the Nano GND.
- **Sketch:** upload `verociti_board.ino` unchanged (`BOARD 'A'`). D8–D13 and A0–A5 stay free for later.
- **Calibrate:** click the four road tips in order: **north, east, south, west**.

**Two junctions (J1 + J3):** extend the east road of J1 by one more crossing, ~25 cm on, with its own north, south
and east arms. Wire J3's heads exactly like J1's, but to **A0 A1 A2** (EW) and **A3 A4 A5** (NS), on the same Nano.
Switch *Board layout* to *Two junctions (J1 + J3)* and calibrate on J1's north tip, J3's east tip, J3's south tip
and J1's west tip. A red box on J1's west arm turns both junctions green toward the east.

To grow to the five-junction board later, switch *Board layout* to *Five junctions* and follow the rest of this guide.

## 1. Parts

| Part | Qty |
|---|---|
| Arduino Nano (clone OK) + USB **data** cable | 2 (+1 spare) |
| Traffic-light LED module (pins R, Y, G, GND) | 20 (+3 spare) |
| Breadboard | 2 |
| Female-to-male jumpers (30 cm if possible) | ~100 |
| Male-to-male jumpers | ~40 |
| Thin hookup wire | 3–5 m |
| MDF/ply ~90 × 90 cm, strips for a frame so wiring fits underneath | 1 |
| **Matte** black chart paper/vinyl (roads), white marker/tape (lanes), yellow tape (edges) | |
| Phone + overhead mount (ring-light stand with phone arm, or wooden gantry) | 1 |
| Small **white** boxes (cars), 1 **red** box (ambulance) | 12–15 + 1 |

No 12 V supply, no shift registers, no capacitors. Everything runs from the laptop's USB.

## 2. Board layout

A 3×3 grid of roads. Only the plus is signalled: J2 north, J1 west, J3 centre, J4 east, J5 south.
The four grid corners are plain bends with no lights. Only the middle arms leave the grid.

```
                 │
         ┌────── J2 ──────┐
         │       │        │
   ──── J1 ───── J3 ───── J4 ────
         │       │        │
         └────── J5 ──────┘
                 │
```

Suggested sizes (the software only needs the proportions):
- **22 cm** between neighbouring junctions (J1→J3, J3→J4, …)
- roads **7.5 cm** wide, junction boxes ~8 cm square
- outer arms up to 22 cm long (shorter is fine if the board is small)
- 4 signal heads per junction, one at each corner of the box, facing the incoming road

Keep the roads **matte black** and the board evenly lit. Glare is the main thing that confuses the camera.

## 3. Wiring

The two heads facing each other along a road always show the same light, so each **pair** shares
one pin: join the two modules' R wires together, the Y wires together and the G wires together.
(EW pair = the heads for traffic moving east/west. NS pair = the heads for north/south traffic.)

**Nano A: J1, J2, J3**

| Junction | EW R / Y / G | NS R / Y / G |
|---|---|---|
| J1 | D2 / D3 / D4 | D5 / D6 / D7 |
| J2 | D8 / D9 / D10 | D11 / D12 / D13 |
| J3 | A0 / A1 / A2 | A3 / A4 / A5 |

**Nano B: J4, J5**

| Junction | EW R / Y / G | NS R / Y / G |
|---|---|---|
| J4 | D2 / D3 / D4 | D5 / D6 / D7 |
| J5 | D8 / D9 / D10 | D11 / D12 / D13 |

- Every module's **GND** goes to its Nano's GND (use a breadboard rail as the shared GND).
- Don't use D0/D1 (USB serial) or A6/A7 (input-only on a Nano).
- If a Nano gets warm or resets, wire only one head per pin (the one facing the audience).

## 4. Flash the Nanos

1. Install the Arduino IDE and open `hardware/verociti_board/verociti_board.ino`.
2. Tools → Board → **Arduino Nano**. Clones: Tools → Processor → **ATmega328P (Old Bootloader)**.
   CH340 clones may need the CH340 driver on Windows.
3. Leave `#define BOARD 'A'` and upload to the first Nano.
4. Change it to `#define BOARD 'B'` and upload to the second Nano.
5. On power-up every head blinks red → amber → green, then flashes amber until the laptop talks to it.
   That's your first wiring check.

If lamps light when they should be off (and the other way round), your modules are common-VCC:
set `#define ACTIVE_HIGH 0` and re-upload.

## 5. Run it

1. `pip install -r "city flow model/requirements.txt"` (adds `pyserial`), then `start.bat` as usual.
2. Open the dashboard → Traffic portal → **Physical Board**.
3. **LED boards:** plug in both Nanos. "Auto-detect all Nanos" finds them (the chip shows `2 Nanos`).
   Switch the mode to **Lamp test**: every colour on every head in turn, one group at a time.
   Any head that lights out of sequence is mis-wired.
4. **Camera:** install **IP Webcam** (Android) on the phone. Connect the phone to the **laptop's hotspot**
   (not college WiFi), set the resolution to 1920×1080, and tap *Start server*. It shows an address like
   `http://192.168.137.23:8080`. Enter `http://<that-ip>:8080/video` as the camera source and click **Connect**.
5. **Calibrate:** click *Calibrate*, then click the centre of the four corner bends in order:
   **NW, NE, SE, SW** (north = the J2 side).
6. With **no cars on the board**, click **Capture empty board**. Detection then ignores lane paint, glare and the LEDs.
7. Switch the mode back to **AI from camera**.

Settings are saved in `city flow model/board_config.json`, so a restart doesn't need recalibrating
unless the phone moves.

## 6. Cars and plates

- White boxes about 3–4 cm long. The software learns their size, so a nose-to-tail line of cars is still counted correctly.
- Print plates **big and bold** on the top face (text about 1.5 cm tall), using real state codes:
  `TN07`, `KA01`, `MH12`, `DL03`, `GJ05`, `WB20`, `OD02`, …
  A reading that doesn't start with a real state code is rejected as noise.
- The **red** box is the ambulance.

## 7. Demo script (about 3 minutes)

1. **"The AI controls every light."** Point at the *Sent to Nanos* line on the dashboard and the AI's reason under each junction.
2. **Queue.** Put 3–4 cars on J3's north approach. Within a few seconds J3 goes amber → green for NS,
   and the reason reads "Higher load on NS". Slide the cars through, and the queue clears and the lights move on.
3. **Plates.** Point at the vehicle table: plates read live from the camera.
4. **Ambulance.** Drop the red box on J1's west arm. J1 and J3 go amber → all-red → green in its direction
   ("Holding GREEN for Ambulance Corridor"). Lift it off and the most-starved approach is served first (recovery).
5. **Fail-safe.** Cover the camera. The AI rung steps down to fixed-time on its own and the board keeps cycling safely.
   Unplug a Nano's USB and its heads flash amber.

## 8. If something goes wrong on the day

| Symptom | Fix |
|---|---|
| Chip says "No Arduino found" | Data cable, not a charge cable. Try the other USB port. Install the CH340 driver. |
| All heads flash amber | Nano is powered but not hearing the laptop: check the LED-boards chip, or pick the COM port manually. |
| Camera chip red | Phone and laptop on the same hotspot? Open `http://<ip>:8080` in a browser to check. |
| Cars not detected / ghost cars | Recalibrate, then *Capture empty board* again with the board empty. Kill glare. |
| Anything unfixable | Switch the mode to **Mirror simulation**: the LEDs follow the software simulation, and the dashboard still tells the story. |

Test without any hardware: set the camera source to `synthetic` and click roads in the feed to drop cars and the ambulance.
Automated tests: `python "city flow model/scripts/test_board_pipeline.py"`.

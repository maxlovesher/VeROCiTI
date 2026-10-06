/*
  VeROCiTI demo board — LED driver for one Arduino Nano.
  =======================================================
  This board makes no decisions. The Signal AI on the laptop decides every
  phase and sends one line per tick over USB:

      S<J1 EW><J1 NS><J2 EW><J2 NS>...<J5 EW><J5 NS>\n     e.g. SGRRGYRGRRG

  each character R / Y / G (that head group's lamp) or O (off, lamp test).
  Two Nanos share the board and both receive the same line:
      Nano A -> J1, J2, J3   (18 pins)
      Nano B -> J4, J5       (12 pins)

  Fail-safe: if nothing arrives for 3 s, every head flashes amber — what a
  real controller does when it loses its brain.

  BEFORE UPLOADING: set BOARD below to 'A' or 'B'.
  Arduino IDE: Tools > Board > Arduino Nano. Clones usually need
  Tools > Processor > ATmega328P (Old Bootloader).
*/

#define BOARD 'A'          // <-- 'A' for the Nano driving J1-J3, 'B' for J4-J5

// Traffic-light modules with a shared GND pin light on HIGH. If yours share
// VCC instead (lamps come on when the pin is LOW), set this to 0.
#define ACTIVE_HIGH 1

const unsigned long FAILSAFE_MS = 3000;
const unsigned long FLASH_MS = 500;

#if BOARD == 'A'
const uint8_t FIRST_GROUP = 0;           // frame chars 0..5 = J1 EW, J1 NS, J2 EW, J2 NS, J3 EW, J3 NS
const uint8_t GROUPS = 6;
// {R, Y, G} per head group, in frame order.
const uint8_t PINS[GROUPS][3] = {
  {2, 3, 4},       // J1 EW
  {5, 6, 7},       // J1 NS
  {8, 9, 10},      // J2 EW
  {11, 12, 13},    // J2 NS
  {A0, A1, A2},    // J3 EW
  {A3, A4, A5},    // J3 NS
};
#define HELLO "VEROCITI BOARD A J1-J3"
#else
const uint8_t FIRST_GROUP = 6;           // frame chars 6..9 = J4 EW, J4 NS, J5 EW, J5 NS
const uint8_t GROUPS = 4;
const uint8_t PINS[GROUPS][3] = {
  {2, 3, 4},       // J4 EW
  {5, 6, 7},       // J4 NS
  {8, 9, 10},      // J5 EW
  {11, 12, 13},    // J5 NS
};
#define HELLO "VEROCITI BOARD B J4-J5"
#endif

char line[24];
uint8_t lineLen = 0;
unsigned long lastFrameMs = 0;

void writeLamp(uint8_t pin, bool on) {
  digitalWrite(pin, (on == (ACTIVE_HIGH == 1)) ? HIGH : LOW);
}

void showGroup(uint8_t g, char state) {
  writeLamp(PINS[g][0], state == 'R');
  writeLamp(PINS[g][1], state == 'Y');
  writeLamp(PINS[g][2], state == 'G');
}

void showAll(char state) {
  for (uint8_t g = 0; g < GROUPS; g++) showGroup(g, state);
}

void applyFrame() {
  // line = "S" + 10 state chars; this board only reads its own slice.
  if (lineLen < 1 + FIRST_GROUP + GROUPS || line[0] != 'S') return;
  for (uint8_t g = 0; g < GROUPS; g++) {
    char c = line[1 + FIRST_GROUP + g];
    if (c != 'R' && c != 'Y' && c != 'G' && c != 'O') return;   // corrupted line: keep the last good state
  }
  for (uint8_t g = 0; g < GROUPS; g++) showGroup(g, line[1 + FIRST_GROUP + g]);
  lastFrameMs = millis();
}

void setup() {
  for (uint8_t g = 0; g < GROUPS; g++)
    for (uint8_t c = 0; c < 3; c++) pinMode(PINS[g][c], OUTPUT);

  // Power-on self test: red, amber, green on every head.
  showAll('R'); delay(300);
  showAll('Y'); delay(300);
  showAll('G'); delay(300);
  showAll('R');

  Serial.begin(115200);
  Serial.println(F(HELLO));
  lastFrameMs = millis();
}

void loop() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (lineLen) {
        line[lineLen] = '\0';
        if (line[0] == '?') Serial.println(F(HELLO));
        else applyFrame();
      }
      lineLen = 0;
    } else if (lineLen < sizeof(line) - 1) {
      line[lineLen++] = ch;
    } else {
      lineLen = 0;   // overlong garbage: drop it
    }
  }

  if (millis() - lastFrameMs > FAILSAFE_MS) {
    showAll((millis() / FLASH_MS) % 2 == 0 ? 'Y' : 'O');
  }
}

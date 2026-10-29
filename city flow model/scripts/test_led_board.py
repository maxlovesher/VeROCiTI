"""
test_led_board.py — offline tests for led_board.py (no Arduino, no camera, no server).

    python "city flow model/scripts/test_led_board.py"
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import led_board as lb


class FakeLink:
    def __init__(self):
        self.frames = []
        self.setting = "auto"
        self.connected = True
        self.port = "COM9"
        self.error = None
        self.frames_sent = 0

    def send(self, frame):
        self.frames.append(frame)

    def set_port(self, setting):
        self.setting = setting

    @staticmethod
    def list_ports():
        return []


class FakeAgent:
    phase_names = ["EW", "NS"]

    def __init__(self, current_phase=0, is_yellow=False, is_all_red=False):
        self.current_phase, self.is_yellow, self.is_all_red = current_phase, is_yellow, is_all_red


def make_board(sim=None, swapped=None, exchanged=False):
    path = os.path.join(tempfile.mkdtemp(), "cfg.json")
    board = lb.LedBoard(sim_signals=sim, link=FakeLink(), config_path=path, start=False)
    board.exchanged = exchanged                       # most tests want the wire in plain junction order
    if swapped is not None:
        board.swapped = set(swapped)
    return board


SIM_EW_GREEN = lambda: {"J1": {"EW": "G", "NS": "R"}, "J3": {"EW": "G", "NS": "R"}}


class TestFrames(unittest.TestCase):
    def test_frame_has_ten_lamp_characters_in_junction_order(self):
        sig = lb.all_signals("R")
        sig["J1"]["EW"], sig["J3"]["NS"] = "G", "Y"
        self.assertEqual(lb.encode_frame(sig), "SGRRRRYRRRR\n")           # J1 EW, J1 NS, J2 EW, J2 NS, J3 EW, J3 NS, ...

    def test_unknown_lamps_go_dark_rather_than_corrupting_the_frame(self):
        self.assertEqual(lb.encode_frame({"J1": {"EW": "X", "NS": "G"}}), "SOGOOOOOOOO\n")

    def test_swapped_junction_trades_ew_and_ns_on_the_wire_only(self):
        sig = {j: {"EW": "G", "NS": "R"} for j in lb.JUNCTIONS}
        wire = lb.to_wire(sig, {"J1"})
        self.assertEqual(wire["J1"], {"EW": "R", "NS": "G"})
        self.assertEqual(wire["J3"], {"EW": "G", "NS": "R"})
        self.assertEqual(sig["J1"], {"EW": "G", "NS": "R"})        # the logical state is untouched

    def test_exchanged_junctions_use_each_others_pins_and_nothing_else_changes(self):
        sig = lb.all_signals("O")
        sig["J1"], sig["J3"] = {"EW": "G", "NS": "R"}, {"EW": "R", "NS": "G"}
        wire = lb.to_wire(sig, set(), exchanged=True)
        self.assertEqual(wire["J1"], {"EW": "R", "NS": "G"})       # J3's state on J1's pins
        self.assertEqual(wire["J3"], {"EW": "G", "NS": "R"})
        self.assertEqual(wire["J2"], {"EW": "O", "NS": "O"})
        self.assertEqual(lb.encode_frame(wire), "SRGOOGROOOO\n")
        self.assertEqual(lb.encode_frame(lb.to_wire(sig, set())), "SGROORGOOOO\n")      # not exchanged: untouched

    def test_exchange_and_swap_combine_per_logical_junction(self):
        sig = lb.all_signals("O")
        sig["J1"] = {"EW": "G", "NS": "R"}
        wire = lb.to_wire(sig, {"J1"}, exchanged=True)
        self.assertEqual(wire["J3"], {"EW": "R", "NS": "G"})       # J1's heads, crossed, on J3's pins

    def test_agent_signals_mirror_the_ai(self):
        self.assertEqual(lb.agent_signals(FakeAgent(0)), {"EW": "G", "NS": "R"})
        self.assertEqual(lb.agent_signals(FakeAgent(1)), {"EW": "R", "NS": "G"})
        self.assertEqual(lb.agent_signals(FakeAgent(0, is_yellow=True)), {"EW": "Y", "NS": "R"})
        self.assertEqual(lb.agent_signals(FakeAgent(0, is_all_red=True)), {"EW": "R", "NS": "R"})


class TestPatterns(unittest.TestCase):
    def test_wiring_check_puts_the_two_junctions_in_opposite_states(self):
        sig = lb.wiring_check()
        self.assertEqual(sig["J1"], {"EW": "G", "NS": "R"})
        self.assertEqual(sig["J3"], {"EW": "R", "NS": "G"})
        self.assertEqual(sig["J2"], {"EW": "O", "NS": "O"})        # junctions that aren't built stay dark

    def test_lamp_test_lights_exactly_one_lamp_at_a_time(self):
        seen = set()
        for step in range(12):
            sig = lb.lamp_test(step * 0.7 + 0.01)
            lit = [(j, d, c) for j in lb.JUNCTIONS for d, c in sig[j].items() if c != "O"]
            self.assertEqual(len(lit), 1)
            seen.add(lit[0])
        self.assertEqual(len(seen), 12)                            # 2 junctions x 2 pairs x 3 colours

    def test_all_on_lights_every_pair(self):
        board = make_board()
        board.set_mode("allon")
        board.tick(0)
        self.assertEqual(board.link.frames[-1], "SWWWWWWWWWW\n")


class TestController(unittest.TestCase):
    def test_follow_mirrors_the_simulation_through_the_wiring_swap(self):
        board = make_board(sim=SIM_EW_GREEN, swapped={"J1"})
        board.tick(0)
        self.assertEqual(board.signals["J1"], {"EW": "G", "NS": "R"})      # what the AI said
        self.assertEqual(board.link.frames[-1], "SRGOOGROOOO\n")           # J1 crossed on the wire, J3 straight, J2/J4/J5 dark

    def test_a_fresh_clone_with_no_config_drives_the_built_board_correctly(self):
        board = lb.LedBoard(sim_signals=lambda: {"J1": {"EW": "G", "NS": "R"}, "J3": {"EW": "R", "NS": "G"}},
                            link=FakeLink(), config_path=os.path.join(tempfile.mkdtemp(), "none.json"), start=False)
        self.assertEqual(board.swapped, set())
        self.assertTrue(board.exchanged)
        board.tick(0)
        self.assertEqual(board.link.frames[-1], "SRGOOGROOOO\n")         # J1's state on the J3 pins and the reverse, no pair crossed

    def test_a_failing_simulation_sends_nothing_new(self):
        def boom():
            raise RuntimeError("sim down")
        board = make_board(sim=boom)
        board.tick(0)
        self.assertEqual(board.link.frames, [])
        self.assertEqual(board.sim_error, "sim down")

    def test_missing_junction_in_the_simulation_is_reported(self):
        board = make_board(sim=lambda: {"J1": {"EW": "G", "NS": "R"}})
        board.tick(0)
        self.assertIn("J3", board.sim_error)

    def test_manual_starts_from_what_is_showing_then_changes_one_pair(self):
        board = make_board(sim=SIM_EW_GREEN, swapped=set())
        board.tick(0)
        board.set_manual("J3", "NS", "G")
        board.tick(1)
        self.assertEqual(board.mode, "manual")
        self.assertEqual(board.signals["J3"], {"EW": "G", "NS": "G"})
        self.assertEqual(board.signals["J1"], {"EW": "G", "NS": "R"})

    def test_returning_to_follow_hands_control_back_to_the_ai(self):
        board = make_board(sim=SIM_EW_GREEN, swapped=set())
        board.set_manual("*", "*", "O")
        board.tick(0)
        self.assertEqual(board.link.frames[-1], "SOOOOOOOOOO\n")
        board.set_mode("follow")
        board.tick(1)
        self.assertEqual(board.link.frames[-1], "SGROOGROOOO\n")

    def test_rejects_bad_input(self):
        board = make_board()
        with self.assertRaises(ValueError):
            board.set_mode("calibrate")
        with self.assertRaises(ValueError):
            board.set_manual("J9", "EW", "G")
        with self.assertRaises(ValueError):
            board.set_manual("J1", "UP", "G")
        with self.assertRaises(ValueError):
            board.set_manual("J1", "EW", "Z")
        with self.assertRaises(ValueError):
            board.set_swapped("J9", True)

    def test_wiring_and_port_survive_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "cfg.json")
        first = lb.LedBoard(link=FakeLink(), config_path=path, start=False)
        first.set_swapped("J1", False)
        first.set_swapped("J3", True)
        first.set_port("COM7")
        first.set_exchanged(True)
        second = lb.LedBoard(link=FakeLink(), config_path=path, start=False)
        self.assertEqual(second.swapped, {"J3"})
        self.assertTrue(second.exchanged)
        self.assertTrue(second.state()["exchanged"])

    def test_saving_wiring_keeps_the_cameras_settings_in_the_same_file(self):
        import json
        path = os.path.join(tempfile.mkdtemp(), "cfg.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"camera": "http://phone:8080/video", "calibration": [[0.1, 0.2]] * 4}, f)
        board = lb.LedBoard(link=FakeLink(), config_path=path, start=False)
        board.set_swapped("J3", True)
        with open(path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["camera"], "http://phone:8080/video")
        self.assertEqual(len(saved["calibration"]), 4)
        self.assertIn("J3", saved["swapped"])

    def test_state_for_the_dashboard(self):
        board = make_board(sim=SIM_EW_GREEN)
        board.set_mode("wiring")
        board.tick(0)
        st = board.state()
        self.assertEqual(st["mode"], "wiring")
        self.assertEqual(st["junctions"], ["J1", "J3"])
        self.assertEqual(st["signals"]["J3"]["NS"], "G")
        self.assertTrue(st["wiring_hint"])
        self.assertTrue(st["link"]["connected"])
        self.assertTrue(st["frame"].startswith("S"))


class TestUnoLink(unittest.TestCase):
    def test_off_never_opens_a_port_and_a_fresh_link_is_disconnected(self):
        link = lb.UnoLink(setting="off", start=False)
        self.assertFalse(link.connected)
        link.send("SRRRRRRRRRR\n")
        link.close()                                               # safe to call with nothing open
        self.assertFalse(link.connected)

    def test_a_named_port_is_used_as_is(self):
        link = lb.UnoLink(setting="COM7", start=False)
        self.assertEqual(link._pick_port(), "COM7")


if __name__ == "__main__":
    unittest.main(verbosity=2)

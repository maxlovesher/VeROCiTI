"""
test_board_pipeline.py — the physical demo board, end to end, with no hardware.

A rendered synthetic board stands in for the phone camera, and the serial
link is set to "off", so these check the whole chain — camera frame ->
calibration -> car detection -> road/junction assignment -> Signal AI ->
the exact line the Nanos would receive — on any machine, for both the
one-junction and the five-junction board.
Run with:
    python "city flow model/scripts/test_board_pipeline.py"
"""

import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.dirname(HERE)
sys.path.insert(0, SERVER_DIR)

os.environ["BOARD_SERIAL_PORT"] = "off"

import board_layout as bl  # noqa: E402
from board_api import BoardController, lamp_test, to_hardware  # noqa: E402
from board_serial import encode_frame  # noqa: E402
from board_vision import (  # noqa: E402
    BoardVision, SyntheticBoard, normalise_plate, parse_plate_list, rectify_label, snap_to_registered,
)
import cv2  # noqa: E402
import numpy as np  # noqa: E402

FULL = bl.LAYOUTS["full"]
SINGLE = bl.LAYOUTS["single"]
DOUBLE = bl.LAYOUTS["double"]


def controller(layout="full"):
    cfg = os.path.join(tempfile.mkdtemp(prefix="verociti_board_"), "board_config.json")
    board = BoardController(ocr=False, config_path=cfg, start=False)
    board.set_layout(layout)
    board.set_camera(None)
    board.vision.set_corners(board.synthetic.true_points)
    return board


_CLOCK = [time.time()]


def settle(vision, frame, frames=4, step=0.35):
    """Run one camera frame through the vision stage a few times on a simulated clock,
    long enough for detections to pass the confirmation time."""
    out = []
    for _ in range(frames):
        _CLOCK[0] += step
        out = vision.process(frame, now=_CLOCK[0])
    return out


def see(board, cars):
    """Put cars on the synthetic board and push one camera frame through the vision stage."""
    board.synthetic.set_cars(cars)
    frame = board.synthetic.render()
    board.vehicles = settle(board.vision, frame)
    board.camera._publish(frame, 0.0)          # stamped after processing, so a slow machine can't make it look stale
    return board.vehicles


def run(board, ticks, cars):
    for _ in range(ticks):
        see(board, cars)
        board.tick()


def vision_for(layout):
    sb = SyntheticBoard(layout)
    v = BoardVision(layout, ocr=False)
    v.set_corners(sb.true_points)
    return sb, v


class TestLayout(unittest.TestCase):
    def test_full_roads_map_to_their_junction_and_phase(self):
        self.assertEqual(FULL.locate(0, 300)["road"], "road_VW1_J1")
        self.assertEqual(FULL.locate(250, 300)["junction"], "J3")      # nearer J3 than J1
        self.assertEqual(FULL.locate(150, 300)["junction"], "J1")
        self.assertEqual(FULL.locate(300, 420)["phase"], "NS")
        self.assertEqual(FULL.locate(300, 300), {"in_junction": "J3"})
        self.assertIsNone(FULL.locate(200, 200))                       # off-road (the MDF)

    def test_single_only_knows_j1_and_its_four_arms(self):
        self.assertEqual(SINGLE.active, ("J1",))
        self.assertEqual(SINGLE.locate(0, 300)["road"], "road_VW1_J1")
        self.assertEqual(SINGLE.locate(250, 300)["road"], "road_J3_J1")  # the east arm, all of it feeds J1
        self.assertEqual(SINGLE.locate(100, 450)["road"], "road_VN1_J1")
        self.assertEqual(SINGLE.locate(100, 150)["road"], "road_VS1_J1")
        self.assertEqual(SINGLE.locate(100, 300), {"in_junction": "J1"})
        self.assertIsNone(SINGLE.locate(300, 420))                      # J3's road isn't on this board

    def test_double_knows_j1_j3_and_splits_the_link_between_them(self):
        self.assertEqual(DOUBLE.active, ("J1", "J3"))
        self.assertEqual(DOUBLE.locate(0, 300)["road"], "road_VW1_J1")
        self.assertEqual(DOUBLE.locate(160, 300)["road"], "road_J3_J1")   # link, nearer J1
        self.assertEqual(DOUBLE.locate(250, 300)["road"], "road_J1_J3")   # link, nearer J3
        self.assertEqual(DOUBLE.locate(420, 300)["road"], "road_J4_J3")   # J3's east arm
        self.assertEqual(DOUBLE.locate(300, 430)["road"], "road_J2_J3")   # J3's north arm
        self.assertEqual(DOUBLE.locate(300, 170)["road"], "road_J5_J3")   # J3's south arm
        self.assertIsNone(DOUBLE.locate(500, 420))                         # J4's roads aren't on this board

    def test_straight_ahead(self):
        self.assertEqual(FULL.next_junction_straight("road_J1_J3"), "J4")
        self.assertEqual(FULL.next_junction_straight("road_VW1_J1"), "J3")
        self.assertIsNone(FULL.next_junction_straight("road_J3_J4"))
        self.assertIsNone(SINGLE.next_junction_straight("road_VW1_J1"))
        self.assertEqual(DOUBLE.next_junction_straight("road_VW1_J1"), "J3")
        self.assertIsNone(DOUBLE.next_junction_straight("road_J1_J3"))


class TestVision(unittest.TestCase):
    def test_full_detects_cars_and_ambulance_on_the_right_roads(self):
        sb, v = vision_for(FULL)
        sb.set_cars([
            {"x": 250, "y": 300, "plate": "TN07", "kind": "car"},
            {"x": 215, "y": 300, "plate": "KA01", "kind": "car"},     # nose to tail with TN07
            {"x": 300, "y": 420, "plate": "GJ05", "kind": "car"},
            {"x": 0, "y": 300, "plate": "MH12", "kind": "car"},
            {"x": 300, "y": 180, "kind": "ambulance"},
        ])
        out = settle(v, sb.render())
        cars = sorted(d["road"] for d in out if d["kind"] == "car" for _ in range(d["count"]))
        self.assertEqual(cars, ["road_J1_J3", "road_J1_J3", "road_J3_J2", "road_VW1_J1"])
        amb = [d for d in out if d["kind"] == "ambulance"]
        self.assertEqual([(a["junction"], a["phase"]) for a in amb], [("J5", "NS")])

    def test_single_detects_cars_on_each_arm(self):
        sb, v = vision_for(SINGLE)
        sb.set_cars([
            {"x": 0, "y": 300, "plate": "TN07", "kind": "car"},
            {"x": 220, "y": 300, "plate": "KA01", "kind": "car"},
            {"x": 100, "y": 430, "plate": "GJ05", "kind": "car"},
            {"x": 100, "y": 170, "kind": "ambulance"},
        ])
        out = settle(v, sb.render())
        cars = sorted(d["road"] for d in out if d["kind"] == "car")
        self.assertEqual(cars, ["road_J3_J1", "road_VN1_J1", "road_VW1_J1"])
        amb = [d for d in out if d["kind"] == "ambulance"]
        self.assertEqual([(a["road"], a["phase"]) for a in amb], [("road_VS1_J1", "NS")])

    def test_double_detects_cars_on_both_junctions(self):
        sb, v = vision_for(DOUBLE)
        sb.set_cars([
            {"x": 0, "y": 300, "plate": "TN07", "kind": "car"},
            {"x": 250, "y": 300, "plate": "KA01", "kind": "car"},
            {"x": 300, "y": 430, "plate": "GJ05", "kind": "car"},
            {"x": 100, "y": 170, "plate": "MH12", "kind": "car"},
        ])
        out = settle(v, sb.render())
        got = sorted((d["junction"], d["road"]) for d in out)
        self.assertEqual(got, [("J1", "road_VS1_J1"), ("J1", "road_VW1_J1"), ("J3", "road_J1_J3"), ("J3", "road_J2_J3")])

    def test_empty_board_detects_nothing(self):
        for layout in (FULL, SINGLE, DOUBLE):
            sb, v = vision_for(layout)
            self.assertEqual(settle(v, sb.render()), [], layout.name)

    def test_empty_board_reference_still_finds_cars(self):
        sb, v = vision_for(FULL)
        v.capture_reference(sb.render())
        sb.set_cars([{"x": 0, "y": 300, "plate": "MH12", "kind": "car"}])
        self.assertEqual([d["road"] for d in settle(v, sb.render())], ["road_VW1_J1"])

    def test_exposure_jump_after_reference_makes_no_ghost_cars(self):
        sb, v = vision_for(DOUBLE)
        v.capture_reference(sb.render())
        brighter = np.clip(sb.render().astype(np.float32) * 1.2 + 6, 0, 255).astype(np.uint8)   # phone re-exposes
        darker = np.clip(sb.render().astype(np.float32) * 0.8, 0, 255).astype(np.uint8)
        self.assertEqual(settle(v, brighter), [])
        self.assertEqual(settle(v, darker), [])
        sb.set_cars([{"x": 0, "y": 300, "plate": "MH12", "kind": "car"}])
        self.assertEqual([d["road"] for d in settle(v, np.clip(sb.render().astype(np.float32) * 1.2, 0, 255).astype(np.uint8))],
                         ["road_VW1_J1"])                              # a real card still shows up

    def test_one_frame_flicker_is_not_counted(self):
        sb, v = vision_for(DOUBLE)
        empty = sb.render()
        sb.set_cars([{"x": 200, "y": 300, "kind": "ambulance"}, {"x": 0, "y": 300, "plate": "TN07", "kind": "car"}])
        ghost = sb.render()
        _CLOCK[0] += 0.4
        self.assertEqual(v.process(ghost, now=_CLOCK[0]), [])          # seen once: not yet trusted
        self.assertEqual(settle(v, empty), [])                          # gone again: never counted
        self.assertEqual(len(settle(v, ghost)), 2)                      # stays put: counted

    def test_plate_cleanup(self):
        self.assertEqual(normalise_plate("TN O7"), "TN07")
        self.assertEqual(normalise_plate("MHI2"), "MH12")
        self.assertEqual(normalise_plate("0D02"), "OD02")
        self.assertEqual(normalise_plate("KA0I"), "KA01")
        self.assertIsNone(normalise_plate("HELLO"))

    def test_registered_plates(self):
        self.assertEqual(parse_plate_list(["tn07", "KA-01", "junk", "TN07"]), ["TN07", "KA01"])
        reg = ["TN07", "KA01", "MH12"]
        self.assertEqual(snap_to_registered("KA01", reg), ("KA01", 1.0))
        self.assertEqual(snap_to_registered("KA07", reg), ("KA01", 0.8))      # the classic 1 -> 7 misread
        self.assertEqual(snap_to_registered("GJ05", reg), ("GJ05", 0.3))      # unknown: kept, down-weighted
        self.assertEqual(snap_to_registered("KA07", []), ("KA07", 1.0))       # no registry: untouched

    def test_rectify_label_turns_a_tilted_box_upright_without_mirroring(self):
        img = np.full((200, 200, 3), 40, np.uint8)
        cv2.rectangle(img, (60, 80), (140, 120), (240, 240, 240), -1)     # 80 x 40 box
        cv2.rectangle(img, (64, 84), (80, 116), (10, 10, 10), -1)         # dark mark on its left end
        rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((100, 100), 30, 1.0), (200, 200), borderValue=(40, 40, 40))
        mask = ((rot[..., 0] > 150) | (rot[..., 0] < 25)).astype(np.uint8) * 255
        c = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
        crop = rectify_label(rot, cv2.minAreaRect(c))
        self.assertGreater(crop.shape[1], crop.shape[0] * 1.6)            # long side horizontal
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        third = g.shape[1] // 3
        left, right = g[:, :third].mean(), g[:, -third:].mean()
        self.assertGreater(abs(left - right), 30)                         # the mark stayed at one end


class TestController(unittest.TestCase):
    def test_queue_on_ns_makes_the_ai_give_j3_ns_green(self):
        board = controller("full")
        run(board, 3, [])
        self.assertEqual(board.signals["J3"], {"EW": "G", "NS": "R"})
        queue = [{"x": 300, "y": y, "plate": "", "kind": "car"} for y in (360, 395, 430, 240, 205)]
        states = []
        for _ in range(40):
            run(board, 1, queue)
            states.append(dict(board.signals["J3"]))
            if board.signals["J3"]["NS"] == "G":
                break
        self.assertEqual(board.signals["J3"], {"EW": "R", "NS": "G"}, f"J3 never served NS: {states}")
        self.assertIn({"EW": "Y", "NS": "R"}, states)      # went through amber on the way
        self.assertIn("NS", board.coordinator.agents["J3"].last_decision_reason)

    def test_single_junction_queue_makes_the_ai_give_j1_ns_green(self):
        board = controller("single")
        run(board, 3, [])
        self.assertEqual(board.signals["J1"], {"EW": "G", "NS": "R"})
        queue = [{"x": 100, "y": y, "plate": "", "kind": "car"} for y in (380, 420, 460, 220, 180)]
        for _ in range(40):
            run(board, 1, queue)
            if board.signals["J1"]["NS"] == "G":
                break
        self.assertEqual(board.signals["J1"], {"EW": "R", "NS": "G"})
        self.assertEqual(board.state()["junctions"]["J1"]["cars"], {"EW": 0, "NS": 5})
        self.assertEqual(encode_frame(board.signals)[1:3], "RG")   # Nano A's D2-D7 slice

    def test_ambulance_preempts_its_junction_then_releases(self):
        board = controller("full")
        run(board, 12, [])
        amb = [{"x": 100, "y": 420, "kind": "ambulance"}]   # on the VN1 arm, queued at J1
        run(board, 8, amb)
        self.assertEqual(board.ambulance, {"J1": "NS"})
        self.assertEqual(board.signals["J1"], {"EW": "R", "NS": "G"})
        run(board, 4, [])
        board.tick(time.time() + 5)                                     # gone for longer than the hold: released
        self.assertEqual(board.ambulance, {})
        self.assertIsNone(board.coordinator.agents["J1"].emergency_override)

    def test_ambulance_corridor_runs_ahead_only_on_the_board(self):
        board = controller("full")
        run(board, 2, [{"x": 250, "y": 300, "kind": "ambulance"}])   # road_J1_J3, heading east
        self.assertEqual(board.ambulance, {"J1": "EW", "J3": "EW", "J4": "EW"})   # behind, ahead, and after
        single = controller("single")
        run(single, 2, [{"x": 0, "y": 300, "kind": "ambulance"}])     # west arm of the lone junction
        self.assertEqual(single.ambulance, {"J1": "EW"})

    def test_double_ambulance_clears_both_junctions_in_its_path(self):
        board = controller("double")
        run(board, 12, [])
        run(board, 8, [{"x": 0, "y": 300, "kind": "ambulance"}])      # J1's west arm, heading east
        self.assertEqual(board.ambulance, {"J1": "EW", "J3": "EW"})
        self.assertEqual(board.signals["J1"]["EW"], "G")
        self.assertEqual(board.signals["J3"]["EW"], "G")
        frame = encode_frame(board.signals)
        self.assertEqual((frame[1], frame[5]), ("G", "G"))            # Nano A: J1 EW on D2-D4, J3 EW on A0-A2

    def test_switching_layout_resets_geometry(self):
        board = controller("full")
        run(board, 2, [{"x": 250, "y": 300, "kind": "ambulance"}])
        board.set_layout("single")
        self.assertEqual(board.ambulance, {})
        self.assertEqual(board.state()["active_junctions"], ["J1"])
        self.assertEqual(board.state()["calibration"]["labels"][0], "J1 box top-left corner")

    def test_camera_loss_steps_down_the_ladder(self):
        board = controller("full")
        for _ in range(30):
            board.tick()
        self.assertFalse(board.data_ok)
        self.assertNotEqual(board.coordinator.degradation.current, "FULL_AI")

    def test_serial_frame_and_lamp_test(self):
        signals = {j: {"EW": "G", "NS": "R"} for j in bl.SIGNALLED}
        signals["J3"] = {"EW": "Y", "NS": "R"}
        self.assertEqual(encode_frame(signals), "SGRGRYRGRGR\n")
        frames = {encode_frame(lamp_test(t * 0.7 + 0.1)) for t in range(13)}
        self.assertIn("SRRRRRRRRRR\n", frames)
        self.assertIn("SGOOOOOOOOO\n", frames)
        self.assertEqual(len(frames), 13)
        single = {encode_frame(lamp_test(t * 0.7 + 0.1, ("J1",))) for t in range(5)}
        self.assertEqual(single, {"SRRRRRRRRRR\n", "SYYYYYYYYYY\n", "SGGGGGGGGGG\n", "SGOOOOOOOOO\n", "SOGOOOOOOOO\n"})

    def test_swapped_junction_sends_ew_and_ns_the_other_way_round(self):
        signals = {j: {"EW": "G", "NS": "R"} for j in bl.SIGNALLED}
        self.assertEqual(encode_frame(to_hardware(signals, set())), "SGRGRGRGRGR\n")
        self.assertEqual(encode_frame(to_hardware(signals, {"J3"})), "SGRGRRGGRGR\n")   # only J3's pair flips
        board = controller("double")
        board.set_swapped("J3", True)
        board.set_mode("manual")
        board.set_manual("J3", "EW", "G")
        board.set_manual("J3", "NS", "R")
        board.tick()
        self.assertEqual(board.signals["J3"], {"EW": "G", "NS": "R"})                     # dashboard meaning unchanged
        self.assertEqual(board.serial._frame[5:7], "RG")                                   # J3 chars on the wire: swapped
        self.assertEqual(board.state()["swapped_junctions"], ["J3"])

    def test_empty_board_reference_survives_a_restart(self):
        board = controller("double")
        board.set_camera("synthetic")
        see(board, [])
        self.assertTrue(board.capture_reference())
        again = BoardController(ocr=False, config_path=board.config_path, start=False)
        self.assertIsNotNone(again.vision.reference)                       # reloaded from disk
        again.set_corners(board.synthetic.true_points[::-1])                # calibration changed...
        third = BoardController(ocr=False, config_path=board.config_path, start=False)
        self.assertIsNone(third.vision.reference)                           # ...so the old shot isn't trusted

    def test_old_road_tip_calibration_is_converted_to_box_corners(self):
        import json
        board = controller("double")
        tips_in_frame = board.synthetic.CAMERA["double"]          # what an old road-tip calibration saved
        cfg = json.load(open(board.config_path))
        cfg.update({"layout": "double", "corners": tips_in_frame, "camera": None})
        cfg.pop("calibration_points", None)                        # old configs didn't record it
        json.dump(cfg, open(board.config_path, "w"))
        again = BoardController(ocr=False, config_path=board.config_path, start=False)
        np.testing.assert_allclose(again.vision.corners, board.synthetic.true_points, atol=0.5)
        self.assertEqual(again.state()["calibration"]["labels"][0], "J1 box top-left corner")
        sb = again.synthetic
        sb.set_cars([{"x": 0, "y": 300, "plate": "TN07", "kind": "car"}, {"x": 300, "y": 430, "plate": "KA01", "kind": "car"}])
        got = sorted(d["road"] for d in settle(again.vision, sb.render()))
        self.assertEqual(got, ["road_J2_J3", "road_VW1_J1"])        # detections land on the same roads

    def test_auto_calibrate_from_one_click_per_junction_box(self):
        board = controller("double")
        board.vision.set_corners(None)
        sb = board.synthetic
        sb.edge_strips = True
        sb.set_layout(board.layout)                                       # redraw with the grey edge strips
        board.camera._publish(sb.render(), 0.0)
        centres = [cv2.perspectiveTransform(np.array([[board.layout.to_canvas(*bl.NODES[j], sb.S)]], np.float32), sb._H)[0, 0].tolist()
                   for j in board.layout.active]
        centres = [[x + 9, y - 7] for x, y in centres]                    # a slightly sloppy click is fine
        result = board.auto_calibrate(centres, settle_s=0)
        self.assertLess(result["fit_error_units"], 3)
        np.testing.assert_allclose(board.vision.corners, sb.true_points, atol=4)
        self.assertTrue(result["empty_board_captured"])
        with self.assertRaises(ValueError):
            board.auto_calibrate(centres[:1], settle_s=0)                 # one click short

    def test_manual_clicks_on_the_road_ends_are_rejected(self):
        from board_vision import calibration_shape_problem
        board = controller("double")
        good = board.synthetic.true_points
        self.assertIsNone(calibration_shape_problem(good, board.layout.calibration_points))
        tips = board.synthetic.CAMERA["double"]                           # what clicking the road ends gives
        self.assertIn("box corners", calibration_shape_problem(tips, board.layout.calibration_points))
        self.assertIn("order", calibration_shape_problem(good[::-1], board.layout.calibration_points))

    def test_ambulance_text_matching(self):
        from board_vision import is_ambulance_text
        for t in ("AMBULANCE", "AMBULANCF", "4MBULANCE", "AMBU LANCE", "AMBULANC", "AMBUIANCE", "AWBULANCE"):
            self.assertTrue(is_ambulance_text(t), t)
        for t in ("TN07", "KA01", "MH01DV5346", "HELLO", "BALANCE", "ADVANCE", "AMB"):
            self.assertFalse(is_ambulance_text(t), t)

    def test_card_reading_ambulance_opens_the_corridor(self):
        try:
            import easyocr  # noqa: F401
        except Exception:
            self.skipTest("EasyOCR not installed")
        cfg = os.path.join(tempfile.mkdtemp(prefix="verociti_board_"), "board_config.json")
        board = BoardController(ocr=True, config_path=cfg, start=False)
        board.set_layout("double")
        board.set_camera(None)
        board.vision.set_corners(board.synthetic.true_points)
        card = {"x": 0, "y": 300, "plate": "AMBULANCE", "kind": "car", "w": 62, "h": 26}   # white card on J1's west road
        board.synthetic.set_cars([card])
        deadline = time.time() + 90                                       # first OCR call loads the model
        seen = []
        while time.time() < deadline:
            frame = board.synthetic.render()
            board.camera._publish(frame, 0.0)
            board.vehicles = board.vision.process(frame)
            board.tick()
            seen = [(v["kind"], v["plate"]) for v in board.vehicles]
            if board.ambulance:
                break
            time.sleep(0.3)
        self.assertEqual(board.ambulance, {"J1": "EW", "J3": "EW"}, f"never became an ambulance: {seen}")
        for _ in range(12):                                              # amber + all-red clearance, then green
            frame = board.synthetic.render()
            board.camera._publish(frame, 0.0)
            board.vehicles = board.vision.process(frame)
            board.tick()
        self.assertEqual(board.signals["J1"]["EW"], "G")
        self.assertEqual(board.signals["J3"]["EW"], "G")
        self.assertEqual(board.state()["junctions"]["J1"]["cars"], {"EW": 0, "NS": 0})   # not counted as traffic

    def test_pink_ambulance_card_is_found_but_a_pink_toy_car_is_not(self):
        sb, v = vision_for(DOUBLE)
        pink = (203, 170, 255)                                          # BGR light pink, like the real card
        sb.set_cars([
            {"x": 200, "y": 300, "plate": "", "kind": "ambulance", "colour": pink, "w": 60, "h": 26},
            {"x": 300, "y": 430, "plate": "TN07", "kind": "car", "body": pink},           # pink car, white label
            {"x": 100, "y": 170, "plate": "KA01", "kind": "car", "body": (40, 40, 220)},  # red car, white label
        ])
        out = settle(v, sb.render())
        amb = [d for d in out if d["kind"] == "ambulance"]
        self.assertEqual(len(amb), 1, out)
        self.assertEqual(amb[0]["phase"], "EW")
        self.assertEqual(sorted(d["road"] for d in out if d["kind"] == "car"), ["road_J2_J3", "road_VS1_J1"])

    def test_corridor_covers_the_junction_behind_and_holds_while_the_ambulance_stays(self):
        board = controller("double")
        run(board, 12, [])
        amb = [{"x": 250, "y": 300, "kind": "ambulance"}]                # between J1 and J3, nearer J3
        run(board, 10, amb)
        self.assertEqual(board.ambulance, {"J1": "EW", "J3": "EW"})     # both ends of its road
        self.assertEqual((board.signals["J1"]["EW"], board.signals["J3"]["EW"]), ("G", "G"))
        run(board, 15, amb)                                             # still there: still green
        self.assertEqual((board.signals["J1"]["EW"], board.signals["J3"]["EW"]), ("G", "G"))
        now = time.time()
        board._apply_ambulance([], now + 1)                             # a blink in detection: kept
        self.assertEqual(board.ambulance, {"J1": "EW", "J3": "EW"})
        board._apply_ambulance([], now + 5)                             # really gone: released
        self.assertEqual(board.ambulance, {})

    def test_calibration_follows_the_phone_when_it_moves(self):
        board = controller("double")
        sb = board.synthetic
        sb.edge_strips = True
        sb.set_layout(board.layout)
        frame = sb.render()                                               # empty board
        board.camera._publish(frame, 0.0)
        board.set_corners(sb.true_points)                                 # calibrate: remembers this view
        self.assertTrue(board.capture_reference())                        # and the empty-board shot
        self.assertTrue(board.state()["calibration"]["tracking"])
        sb.set_cars([{"x": 0, "y": 300, "plate": "TN07", "kind": "car"}, {"x": 300, "y": 430, "plate": "KA01", "kind": "car"}])
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 7, 1.0)               # the phone gets knocked: rotated 7 degrees
        M[:, 2] += (25, -15)
        moved = cv2.warpAffine(sb.render(), M, (w, h))
        board._track_camera(moved, 10.0)
        board._track_camera(moved, 20.0)                                  # second check confirms the move
        truth = cv2.transform(np.float32(sb.true_points).reshape(-1, 1, 2), M).reshape(-1, 2)
        np.testing.assert_allclose(board.vision.corners, truth, atol=3)
        got = sorted(d["road"] for d in settle(board.vision, moved))
        self.assertEqual(got, ["road_J2_J3", "road_VW1_J1"])              # right roads, and the old empty-board
        self.assertIsNotNone(board.vision.reference)                      # shot still lines up (no strip ghosts)

    def test_sim_mode_mirrors_the_simulation(self):
        board = controller("full")
        mirrored = {j: {"EW": "R", "NS": "G"} for j in bl.SIGNALLED}
        board.sim_signals = lambda: mirrored
        board.set_mode("sim")
        self.assertEqual(board.tick(), mirrored)

    def test_state_shape(self):
        board = controller("full")
        run(board, 2, [{"x": 0, "y": 300, "plate": "MH12", "kind": "car"}])
        st = board.state()
        self.assertTrue(st["calibration"]["calibrated"])
        self.assertTrue(st["camera"]["live"])
        self.assertEqual(st["junctions"]["J1"]["cars"], {"EW": 1, "NS": 0})
        self.assertEqual(set(st["junctions"]), set(bl.SIGNALLED))
        self.assertEqual(st["layout"], "full")
        self.assertEqual(st["vehicles"][0]["plate_state"], None)          # OCR is off in tests
        board.set_registered_plates(["tn07", "ka01"])
        self.assertEqual(board.state()["registered_plates"], ["TN07", "KA01"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_board_camera.py — offline tests for board_camera.py. A synthetic board (dark roads, white cards,
a pink card, glowing LEDs) is drawn flat, then viewed through a tilted "camera", so calibration,
detection, the empty-board comparison, the ambulance corridor and the AI hookup are all exercised
with no phone, no Arduino and no server.

    python "city flow model/scripts/test_board_camera.py"
"""

import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import board_camera as bc

FRAME_W, FRAME_H = 1280, 960
# Where the flat plan's corners land in the "camera": a mild perspective tilt, like a phone not quite overhead.
PLAN_TO_FRAME = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [bc.CANVAS[0], 0], [bc.CANVAS[0], bc.CANVAS[1]], [0, bc.CANVAS[1]]]),
    np.float32([[110, 150], [1180, 120], [1230, 800], [60, 830]]))
PINK = (203, 170, 255)         # BGR light pink, like the real card


def px(x, y):
    cx, cy = bc.to_canvas(x, y)
    return int(round(cx)), int(round(cy))


def plan(cars=(), card=None, leds=(), glow=(), texture=False):
    """The flat board: tan table, dark roads, then cars (white boxes, optionally with a plate), an optional pink card, lit LEDs."""
    img = np.full((bc.CANVAS[1], bc.CANVAS[0], 3), (120, 150, 175), np.uint8)
    if texture:                                          # paper grain and clutter on the table, for the phone-tracker to latch onto
        rng = np.random.RandomState(5)
        noise = cv2.GaussianBlur(rng.randint(-28, 28, img.shape).astype(np.float32), (0, 0), 2.2) * 3.0
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        for _ in range(900):                             # pens, cables, scraps: small hard-edged marks
            x, y = rng.randint(0, bc.CANVAS[0] - 20), rng.randint(0, bc.CANVAS[1] - 20)
            w, h = rng.randint(4, 18), rng.randint(4, 18)
            shade = int(rng.randint(10, 230))
            cv2.rectangle(img, (x, y), (x + w, y + h), (shade, int(rng.randint(10, 230)), int(rng.randint(10, 230))), -1)
    road = (48, 48, 48)
    cv2.rectangle(img, px(bc.VIEW[0], 300 + bc.HALF), px(bc.VIEW[1], 300 - bc.HALF), road, -1)
    for jx in (100, 300):
        cv2.rectangle(img, px(jx - bc.HALF, bc.VIEW[3]), px(jx + bc.HALF, bc.VIEW[2]), road, -1)
    for car in cars:
        x, y = car[0], car[1]
        cv2.rectangle(img, px(x - 15, y + 11), px(x + 15, y - 11), (245, 245, 245), -1)
        if len(car) > 2:                                 # the plate printed on top, big and bold
            cx, cy = px(x, y)
            cv2.putText(img, car[2], (cx - 28, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    if card:
        cv2.rectangle(img, px(card[0] - 30, card[1] + 12), px(card[0] + 30, card[1] - 12), PINK, -1)
    for x, y in glow:                                    # soft red light on the road with no bright core found
        cv2.circle(img, px(x, y), int(14 * bc.SCALE), (40, 40, 200), -1)
    for x, y, colour in leds:                            # a lit LED: saturated glow ringing a blown-out core
        cv2.circle(img, px(x, y), int(9 * bc.SCALE), colour, -1)
        cv2.circle(img, px(x, y), int(2.5 * bc.SCALE), (255, 255, 255), -1)
    return img


def camera_view(plan_img):
    return cv2.warpPerspective(plan_img, PLAN_TO_FRAME, (FRAME_W, FRAME_H), borderValue=(120, 150, 175))


def corners_norm():
    """The four box corners as a person would click them on the camera picture, as fractions of it."""
    out = []
    for p in bc.CAL_WORLD:
        c = np.float32([[bc.to_canvas(*p)]])
        x, y = cv2.perspectiveTransform(c, PLAN_TO_FRAME)[0][0]
        out.append([float(x) / FRAME_W, float(y) / FRAME_H])
    return out


def make_pilot(calibrated=True, with_reference=True, texture=False):
    pilot = bc.CameraPilot(config_path=os.path.join(tempfile.mkdtemp(), "cfg.json"), start=False, ocr=False)
    empty = camera_view(plan(texture=texture))
    pilot.camera._frame, pilot.camera._ts = empty, 1e12
    if calibrated:
        pilot.calibrate(corners_norm())
        if with_reference:
            assert pilot.capture_reference()
    return pilot


def run(pilot, frame, seconds=1.4, t0=1000.0):
    """Feed the same picture for a while so detections get confirmed, as a steady scene would."""
    t = t0
    while t < t0 + seconds:
        pilot._process(frame, t)
        t += 0.1
    return t


class TestGeometry(unittest.TestCase):
    def test_locate_assigns_each_approach_to_the_right_road_and_phase(self):
        cases = {
            (20, 300): ("road_VW1_J1", "J1", "EW"), (160, 300): ("road_J3_J1", "J1", "EW"),
            (240, 300): ("road_J1_J3", "J3", "EW"), (400, 300): ("road_J4_J3", "J3", "EW"),
            (100, 420): ("road_VN1_J1", "J1", "NS"), (100, 180): ("road_VS1_J1", "J1", "NS"),
            (300, 420): ("road_J2_J3", "J3", "NS"), (300, 180): ("road_J5_J3", "J3", "NS"),
        }
        for (x, y), (road, jid, phase) in cases.items():
            w = bc.locate(x, y)
            self.assertEqual((w["road"], w["junction"], w["phase"]), (road, jid, phase), (x, y))

    def test_inside_a_junction_box_has_no_road(self):
        w = bc.locate(105, 295)
        self.assertTrue(w["in_junction"])
        self.assertEqual(w["junction"], "J1")
        self.assertIsNone(w["road"])

    def test_off_the_road_is_nothing(self):
        self.assertIsNone(bc.locate(200, 450))

    def test_canvas_round_trip(self):
        self.assertAlmostEqual(bc.from_canvas(*bc.to_canvas(123.4, 321.0))[0], 123.4, places=6)
        self.assertAlmostEqual(bc.from_canvas(*bc.to_canvas(123.4, 321.0))[1], 321.0, places=6)


class TestCalibrationChecks(unittest.TestCase):
    def good_px(self):
        return [[u * FRAME_W, v * FRAME_H] for u, v in corners_norm()]

    def test_the_real_corners_pass(self):
        self.assertIsNone(bc.calibration_problem(self.good_px()))

    def test_wrong_order_is_rejected_with_a_reason(self):
        self.assertIn("order", bc.calibration_problem(self.good_px()[::-1]))

    def test_a_square_over_the_whole_board_is_rejected(self):
        whole = [[100, 100], [1180, 100], [1180, 860], [100, 860]]
        self.assertIn("doesn't match", bc.calibration_problem(whole))

    def test_points_on_top_of_each_other_are_rejected(self):
        pts = self.good_px()
        pts[1] = [pts[0][0] + 2, pts[0][1] + 2]
        self.assertIsNotNone(bc.calibration_problem(pts))


class TestDetection(unittest.TestCase):
    def test_empty_board_detects_nothing(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan()))
        self.assertEqual(pilot.vehicles, [])

    def test_cars_are_found_on_the_right_roads(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan(cars=[(20, 300), (100, 400), (300, 190), (390, 300)])))
        roads = sorted(v["road"] for v in pilot.vehicles if v["kind"] == "car")
        self.assertEqual(roads, ["road_J4_J3", "road_J5_J3", "road_VN1_J1", "road_VW1_J1"])
        self.assertTrue(all(v["waiting"] for v in pilot.vehicles))

    def test_the_pink_card_is_the_ambulance_and_not_a_car(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan(cars=[(20, 300)], card=(200, 300))))
        kinds = sorted(v["kind"] for v in pilot.vehicles)
        self.assertEqual(kinds, ["ambulance", "car"])
        amb = next(v for v in pilot.vehicles if v["kind"] == "ambulance")
        self.assertEqual(amb["phase"], "EW")

    def test_red_lamp_glow_is_not_an_ambulance_and_a_lit_lamp_is_not_a_car(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan(leds=[(150, 340, (40, 40, 230)), (250, 262, (60, 220, 60))], glow=[(190, 300)])))
        self.assertEqual(pilot.vehicles, [])

    def test_a_single_frame_flicker_is_not_counted(self):
        pilot = make_pilot()
        pilot._process(camera_view(plan(cars=[(20, 300)])), 1000.0)
        self.assertEqual(pilot.vehicles, [])

    def test_the_empty_board_photo_hides_light_coloured_paint(self):
        pilot = make_pilot(with_reference=False)
        painted = plan()
        cv2.rectangle(painted, px(160, 307), px(196, 293), (235, 235, 235), -1)       # a white patch of paint the size of a car
        marked = camera_view(painted)
        run(pilot, marked)
        self.assertEqual(len([v for v in pilot.vehicles if v["kind"] == "car"]), 1)   # without the photo it passes for a car
        pilot.camera._frame = marked
        self.assertTrue(pilot.capture_reference())
        run(pilot, marked, t0=2000.0)
        self.assertEqual(pilot.vehicles, [])

    def test_annotated_view_draws_the_ambulance_label(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan(card=(200, 300))))
        img = pilot.annotated()
        self.assertEqual(img.shape[:2], (bc.CANVAS[1], bc.CANVAS[0]))
        self.assertGreater(int(img[..., 2].astype(int).sum()), 0)


class TestPilot(unittest.TestCase):
    def test_not_calibrated_means_no_data_and_a_safe_fallback(self):
        pilot = make_pilot(calibrated=False)
        for step in range(30):
            pilot.tick(1000.0 + step)
        self.assertNotEqual(pilot.control_mode, "FULL_AI")

    def test_calibration_survives_a_restart_and_a_resolution_change(self):
        pilot = make_pilot()
        again = bc.CameraPilot(config_path=pilot.config_path, start=False, ocr=False)
        self.assertEqual(again.points_norm, pilot.points_norm)
        self.assertIsNotNone(again.reference)
        H_small = again._homography((480, 640))
        H_big = again._homography((960, 1280))
        self.assertEqual(H_small.shape, (3, 3))
        self.assertFalse(np.allclose(H_small, H_big))                  # picture size scales the points, not a fixed pixel list

    def test_bad_corners_are_refused_and_nothing_is_saved(self):
        pilot = make_pilot(calibrated=False)
        with self.assertRaises(ValueError):
            pilot.calibrate([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        self.assertFalse(pilot.calibrated)
        with self.assertRaises(ValueError):
            pilot.calibrate([[0.1, 0.1]])

    def test_reset_clears_calibration_and_the_empty_board_photo(self):
        pilot = make_pilot()
        pilot.reset_calibration()
        self.assertFalse(pilot.calibrated)
        self.assertIsNone(pilot.reference)
        self.assertFalse(os.path.exists(pilot.reference_path))

    def test_a_queue_of_cars_makes_the_ai_give_that_road_green(self):
        pilot = make_pilot()
        queue = camera_view(plan(cars=[(100, 380), (100, 420), (100, 460)]))
        pilot.camera._frame = queue
        t = 1000.0
        for _ in range(60):
            pilot._process(queue, t)
            pilot.tick(t)
            t += 0.9
        j1 = pilot.coordinator.agents["J1"]
        self.assertEqual(j1.phase_names[j1.current_phase], "NS")

    def test_an_ambulance_opens_the_corridor_on_both_junctions_and_releases_it_after(self):
        pilot = make_pilot()
        amb = camera_view(plan(card=(200, 300)))
        pilot.camera._frame = amb
        t = run(pilot, amb)
        pilot.tick(t)
        self.assertEqual(pilot.ambulance, {"J1": "EW", "J3": "EW"})
        self.assertEqual(pilot.coordinator.agents["J1"].emergency_override, "EW")
        gone = camera_view(plan())
        pilot.camera._frame = gone
        run(pilot, gone, t0=t + 0.1)
        pilot.tick(t + 1.0)
        self.assertEqual(pilot.ambulance, {"J1": "EW", "J3": "EW"})   # a blink in detection doesn't drop it
        pilot.tick(t + 6.0)
        self.assertEqual(pilot.ambulance, {})
        self.assertIsNone(pilot.coordinator.agents["J1"].emergency_override)

    def test_an_ambulance_on_a_north_south_arm_only_opens_that_junction(self):
        pilot = make_pilot()
        amb = camera_view(plan(card=(300, 430)))
        pilot.camera._frame = amb
        t = run(pilot, amb)
        pilot.tick(t)
        self.assertEqual(pilot.ambulance, {"J3": "NS"})

    def test_the_corridor_really_shows_green_for_the_ambulance_while_cars_wait_the_other_way(self):
        for card, jid, phase in (((100, 180), "J1", "NS"), ((300, 430), "J3", "NS"), ((200, 300), "J1", "EW")):
            cars = [(20, 300), (400, 300)] if phase == "NS" else [(100, 380), (100, 420)]
            view = camera_view(plan(cars=cars, card=card))
            pilot = make_pilot()
            pilot.camera._frame = view
            t = 1000.0
            for _ in range(80):
                pilot._process(view, t)
                pilot.tick(t)
                t += 0.9
            agent = pilot.coordinator.agents[jid]
            self.assertEqual(pilot.ambulance.get(jid), phase, (card, jid))
            self.assertEqual(agent.phase_names[agent.current_phase], phase, (card, jid, agent.last_decision_reason))
            self.assertFalse(agent.is_yellow or agent.is_all_red, (card, jid))

    def test_state_for_the_dashboard(self):
        pilot = make_pilot()
        run(pilot, camera_view(plan(cars=[(20, 300)])))
        st = pilot.state()
        self.assertTrue(st["calibrated"])
        self.assertTrue(st["has_reference"])
        self.assertEqual(len(st["labels"]), 4)
        self.assertEqual(st["vehicles"][0]["road"], "road_VW1_J1")
        self.assertEqual(st["frame_size"], [FRAME_W, FRAME_H])


class ScriptedReader:
    """Stands in for the OCR: remembers the crops it was sent and answers with whatever the test scripts."""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.sent = []
        self._pending = {}
        self.available = True

    def submit(self, tid, crop):
        self.sent.append((tid, crop))
        if tid in self.answers:
            self._pending[tid] = self.answers[tid]
        return True

    def take_results(self):
        out, self._pending = self._pending, {}
        return out


class TestPlates(unittest.TestCase):
    def test_the_ocr_gets_an_upright_crop_cut_from_the_full_size_picture(self):
        pilot = make_pilot()
        pilot.plates = ScriptedReader()
        run(pilot, camera_view(plan(cars=[(20, 300, "OD02")])), seconds=0.6)
        self.assertTrue(pilot.plates.sent)
        crop = pilot.plates.sent[0][1]
        self.assertEqual(crop.shape[0], 96)                          # LABEL_H tall
        self.assertGreater(crop.shape[1], crop.shape[0])             # long side horizontal

    def test_reads_settle_into_a_plate_on_the_vehicle(self):
        pilot = make_pilot()
        pilot.plates = ScriptedReader()
        frame = camera_view(plan(cars=[(20, 300, "OD02")]))
        run(pilot, frame, seconds=0.5)
        tid = pilot.plates.sent[0][0]
        pilot.plates.answers = {tid: [("OD02", 0.9)]}
        pilot.plates.submit(tid, None)                               # the OCR "finishes" a read
        run(pilot, frame, seconds=1.2, t0=1000.6)
        car = [v for v in pilot.vehicles if v["kind"] == "car"][0]
        self.assertEqual(car["plate"], "OD02")
        self.assertEqual(car["plate_state"], "Odisha")

    def test_a_car_with_a_settled_plate_is_not_read_again(self):
        pilot = make_pilot()
        pilot.plates = ScriptedReader()
        frame = camera_view(plan(cars=[(20, 300, "OD02")]))
        run(pilot, frame, seconds=0.5)
        tid = pilot.plates.sent[0][0]
        pilot._tracks[tid].votes["OD02"] = 2.0                       # two good reads already
        before = len(pilot.plates.sent)
        run(pilot, frame, seconds=2.0, t0=1001.0)
        self.assertEqual(len(pilot.plates.sent), before)

    def test_a_white_card_that_reads_ambulance_becomes_an_ambulance(self):
        pilot = make_pilot()
        pilot.plates = ScriptedReader()
        frame = camera_view(plan(cars=[(200, 300)]))
        run(pilot, frame, seconds=0.5)
        tid = pilot.plates.sent[0][0]
        pilot._tracks[tid].votes["AMBULANCE"] = 0.7
        run(pilot, frame, seconds=1.0, t0=1000.6)
        self.assertEqual([v["kind"] for v in pilot.vehicles], ["ambulance"])
        self.assertIsNone(pilot.vehicles[0]["plate"])

    def test_a_merged_blob_ignores_plate_reads(self):
        pilot = make_pilot()
        pilot.plates = ScriptedReader()
        frame = camera_view(plan(cars=[(20, 300), (50, 300)]))       # two cars touching nose to tail read as one blob
        run(pilot, frame, seconds=0.5)
        tid = pilot.plates.sent[0][0]
        self.assertGreater(pilot._tracks[tid].count, 1)
        pilot.plates.answers = {tid: [("OD02", 0.9)]}
        pilot.plates.submit(tid, None)
        run(pilot, frame, seconds=0.3, t0=1000.6)
        self.assertIsNone(pilot._tracks[tid].plate)

    def test_no_ocr_means_no_plates_but_cars_still_count(self):
        pilot = make_pilot()
        self.assertIsNone(pilot.plates)
        run(pilot, camera_view(plan(cars=[(20, 300)])))
        self.assertIsNone(pilot.vehicles[0]["plate"])
        self.assertIsNone(pilot.state()["plates_available"])


class TestFollowingThePhone(unittest.TestCase):
    NUDGE = cv2.getRotationMatrix2D((FRAME_W / 2, FRAME_H / 2), 1.5, 1.0)
    NUDGE[:, 2] += (18, -12)

    def nudged(self, plan_img):
        return cv2.warpAffine(camera_view(plan_img), self.NUDGE, (FRAME_W, FRAME_H), borderValue=(120, 150, 175))

    def test_the_corners_follow_a_nudged_phone_and_cars_stay_on_their_roads(self):
        pilot = make_pilot(texture=True)
        self.assertIsNotNone(pilot.tracker.anchor)                   # remembered at calibration
        original = np.array(pilot.points_norm) * [FRAME_W, FRAME_H]
        expected = cv2.transform(original.reshape(-1, 1, 2).astype(np.float32), self.NUDGE).reshape(-1, 2)
        moved_view = self.nudged(plan(texture=True))
        pilot._check_motion(moved_view, 1000.0)                      # first look: wait for a second opinion
        self.assertTrue(np.allclose(np.array(pilot.points_norm) * [FRAME_W, FRAME_H], original))
        pilot._check_motion(moved_view, 1002.0)
        now = np.array(pilot.points_norm) * [FRAME_W, FRAME_H]
        self.assertLess(float(np.max(np.linalg.norm(now - expected, axis=1))), 4.0)
        self.assertIsNotNone(pilot.state()["realigned_s_ago"])
        run(pilot, self.nudged(plan(texture=True)))
        self.assertEqual(pilot.vehicles, [])                         # the empty-board photo still fits: no ghost cars
        run(pilot, self.nudged(plan(texture=True, cars=[(20, 300), (300, 190)])), t0=2000.0)
        self.assertEqual(sorted(v["road"] for v in pilot.vehicles), ["road_J5_J3", "road_VW1_J1"])

    def test_a_still_phone_is_left_alone(self):
        pilot = make_pilot(texture=True)
        before = [list(p) for p in pilot.points_norm]
        still = camera_view(plan(texture=True, cars=[(100, 400)]))
        for t in (1000.0, 1002.0, 1004.0):
            pilot._check_motion(still, t)
        self.assertEqual(pilot.points_norm, before)
        self.assertIsNone(pilot.state()["realigned_s_ago"])

    def test_the_snapshot_survives_a_restart_and_reset_forgets_it(self):
        pilot = make_pilot(texture=True)
        again = bc.CameraPilot(config_path=pilot.config_path, start=False, ocr=False)
        self.assertIsNotNone(again.tracker.anchor)
        pilot.reset_calibration()
        self.assertIsNone(pilot.tracker.anchor)
        self.assertFalse(os.path.exists(pilot.anchor_path))

    def test_a_restart_without_a_snapshot_trusts_the_current_view(self):
        pilot = make_pilot(texture=True)
        pilot.tracker.anchor = None
        pilot._check_motion(camera_view(plan(texture=True)), 1000.0)
        self.assertIsNotNone(pilot.tracker.anchor)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
test_board_plates.py — offline tests for board_plates.py (plate cleaning, the AMBULANCE text check, the label crop).
The OCR engine itself isn't run here.

    python "city flow model/scripts/test_board_plates.py"
"""

import os
import sys
import threading
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import board_plates as bp


class TestNormalise(unittest.TestCase):
    def test_letter_digit_lookalikes_are_fixed_by_position(self):
        self.assertEqual(bp.normalise_plate("TN O7"), "TN07")
        self.assertEqual(bp.normalise_plate("MHI2"), "MH12")
        self.assertEqual(bp.normalise_plate("0D02"), "OD02")
        self.assertEqual(bp.normalise_plate("KA0I"), "KA01")

    def test_a_full_plate_keeps_its_tail(self):
        self.assertEqual(bp.normalise_plate("od02ab1234"), "OD02AB1234")
        self.assertEqual(bp.normalise_plate("OD 02 AB 1234"), "OD02AB1234")

    def test_noise_is_rejected(self):
        self.assertIsNone(bp.normalise_plate("HELLO"))
        self.assertIsNone(bp.normalise_plate("XX99"))                # not a state code
        self.assertIsNone(bp.normalise_plate("TN"))
        self.assertIsNone(bp.normalise_plate(""))
        self.assertIsNone(bp.normalise_plate(None))

    def test_an_unreadable_tail_falls_back_to_the_state_and_number(self):
        self.assertEqual(bp.normalise_plate("TN07???"), "TN07")

    def test_state_names(self):
        self.assertEqual(bp.STATE_NAMES["OD"], "Odisha")
        self.assertEqual(bp.STATE_NAMES["PY"], "Puducherry")


class TestAmbulanceText(unittest.TestCase):
    def test_the_word_and_its_usual_misreads(self):
        for text in ("AMBULANCE", "AMBULANCF", "4MBULANCE", "AMBU LANCE", "AMBULANC", "ambulance"):
            self.assertTrue(bp.is_ambulance_text(text), text)

    def test_ordinary_text_and_plates_are_not(self):
        for text in ("BALANCE", "OD02AB1234", "TN07", "AMB", "", None, "LANCE"):
            self.assertFalse(bp.is_ambulance_text(text), text)


class TestRectify(unittest.TestCase):
    def card(self, angle):
        """A card with a big blob at its top-left and a small one at its top-right, drawn tilted."""
        img = np.zeros((400, 500, 3), np.uint8)
        w, h = 200, 80
        local = np.zeros((h, w, 3), np.uint8)
        local[:] = 230
        cv2.circle(local, (22, 20), 14, (0, 0, 0), -1)                # big, top-left
        cv2.circle(local, (w - 22, 20), 6, (0, 0, 0), -1)             # small, top-right
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        M[:, 2] += (250 - w / 2, 200 - h / 2)
        out = cv2.warpAffine(local, M, (500, 400), dst=img, borderMode=cv2.BORDER_TRANSPARENT)
        quad = cv2.transform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), M).reshape(4, 2)
        return out, quad

    def test_the_crop_is_upright_with_the_long_side_horizontal(self):
        for angle in (0, 30, 90, 200):
            img, quad = self.card(angle)
            crop = bp.rectify_label(img, quad)
            self.assertEqual(crop.shape[0], bp.LABEL_H, angle)
            self.assertGreater(crop.shape[1], 2 * crop.shape[0] - 40, angle)

    def test_rotated_never_mirrored(self):
        for angle in (0, 30, 90, 200, 300):
            img, quad = self.card(angle)
            gray = cv2.cvtColor(bp.rectify_label(img, quad), cv2.COLOR_BGR2GRAY)
            dark = gray < 100
            h, w = dark.shape
            quarters = {"tl": dark[:h // 2, :w // 2].sum(), "tr": dark[:h // 2, w // 2:].sum(),
                        "bl": dark[h // 2:, :w // 2].sum(), "br": dark[h // 2:, w // 2:].sum()}
            big = max(quarters, key=quarters.get)
            rest = {k: v for k, v in quarters.items() if k != big and v > 0}
            small = max(rest, key=rest.get)
            # Upright: big top-left with small top-right. Turned 180 degrees: big bottom-right, small bottom-left.
            self.assertIn((big, small), [("tl", "tr"), ("br", "bl")], f"mirrored at {angle} degrees: {quarters}")


class TestReader(unittest.TestCase):
    def test_a_fresh_reader_starts_nothing_until_asked(self):
        reader = bp.PlateReader()
        self.assertIsNone(reader.available)
        self.assertEqual(reader.take_results(), {})

    def test_a_startup_import_clash_is_retried_not_given_up_on(self):
        waits, attempts = [], []

        class Flaky(bp.PlateReader):
            def _load(self):
                attempts.append(1)
                if len(attempts) < 3:
                    raise ImportError("cannot import name 'InterpolationMode' from partially initialized module 'torchvision.transforms'")
                return object()

        original, bp.RETRY_SLEEP = bp.RETRY_SLEEP, waits.append
        try:
            reader = Flaky()
            worker = threading.Thread(target=reader._run, daemon=True)       # after loading it waits for crops, so don't join it
            worker.start()
            worker.join(timeout=1.0)
        finally:
            bp.RETRY_SLEEP = original
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(waits), 2)
        self.assertTrue(reader.available)

    def test_a_real_import_failure_is_reported_after_the_retries(self):
        class Broken(bp.PlateReader):
            def _load(self):
                raise ImportError("no easyocr here")

        original, bp.RETRY_SLEEP = bp.RETRY_SLEEP, lambda s: None
        try:
            reader = Broken()
            reader._run()
        finally:
            bp.RETRY_SLEEP = original
        self.assertFalse(reader.available)
        self.assertFalse(reader.submit(1, np.zeros((96, 200, 3), np.uint8)))      # cars are simply counted without plates

    def test_reading_picks_the_best_valid_plate_and_prefers_ambulance_text(self):
        reader = bp.PlateReader()

        class Fake:
            def __init__(self, found):
                self.found = found

            def readtext(self, img, detail=1, allowlist=None):
                return self.found

        box = lambda x: [[x, 0], [x + 40, 0], [x + 40, 20], [x, 20]]
        reader._reader = Fake([(box(0), "TN", 0.9), (box(50), "07", 0.8)])
        got = dict(reader.read(np.full((96, 200, 3), 255, np.uint8)))
        self.assertIn("TN07", got)
        reader._reader = Fake([(box(0), "AMBULANCE", 0.7)])
        self.assertEqual(reader.read(np.full((96, 200, 3), 255, np.uint8)), [("AMBULANCE", 0.7)])


if __name__ == "__main__":
    unittest.main(verbosity=2)

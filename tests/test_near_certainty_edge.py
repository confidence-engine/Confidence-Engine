import unittest

from scripts.polymarket_bridge import _edge_label, _stance


class TestNearCertaintyEdge(unittest.TestCase):
    def test_near_certainty_in_line_and_stand_aside(self):
        # Both near 1.0 -> in-line edge; stance should be Stand Aside
        edge = _edge_label(0.99, 0.99)
        stance = _stance(edge, "Now")
        self.assertEqual(edge, "in-line")
        self.assertEqual(stance, "Stand Aside")

    def test_small_delta_near_one(self):
        # Within 0.02 tolerance near upper bound should still be in-line
        edge = _edge_label(0.985, 0.999, tol=0.02)
        stance = _stance(edge, "Now")
        self.assertEqual(edge, "in-line")
        self.assertEqual(stance, "Stand Aside")


if __name__ == "__main__":
    unittest.main()

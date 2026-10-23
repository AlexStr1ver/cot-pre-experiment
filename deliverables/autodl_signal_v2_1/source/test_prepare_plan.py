#!/usr/bin/env python3
"""Answer-blind shared-cohort and boundary tests; no model weights/network."""
import copy
import hashlib
import json
import unittest

from prepare_plan import candidate_at, parse_targets, select_common, token_hash, validate_windows


class ToyTokenizer:
    bos_token_id = 100
    eos_token_id = 199
    pieces = {100: "BOS", 101: "Q<think>\n", 1: "a", 2: "\n\n", 3: "b", 4: "c", 5: "d",
              190: "</think>", 191: "<summary>", 192: "</summary>", 199: "EOS"}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[x] for x in ids)

    def convert_tokens_to_ids(self, value):
        return {"</think>": 190, "<summary>": 191, "</summary>": 192}[value]


def record(case, generated):
    return {"case": case, "unique_id": f"dev/{case}", "reference_answer": "GOLD_SECRET",
            "pred_boxed": "UNUSED_PREDICTION", "rounds": [{"prompt": "BOSQ<think>\n",
            "input_token_ids": [100, 101], "generated_token_ids": list(generated)}]}


class PlanChecks(unittest.TestCase):
    def setUp(self):
        self.tok = ToyTokenizer()

    def test_first_boundary_is_inclusive_and_exact(self):
        r = record(1, [1, 2, 3, 2, 4, 190])
        item, reason = candidate_at(r, self.tok, 2, 2)
        self.assertIsNone(reason)
        self.assertEqual(item["cut"], 2)
        self.assertEqual(item["prefix_token_ids_sha256"], token_hash([100, 101, 1, 2]))
        item, _ = candidate_at(r, self.tok, 3, 1)
        self.assertEqual(item["cut"], 4)

    def test_end_markers_exclude_later_boundaries(self):
        for marker in [190, 191, 192, 199]:
            with self.subTest(marker=marker):
                r = record(1, [1, marker, 2])
                self.assertIsNone(candidate_at(r, self.tok, 2, 2)[0])
        # A cut immediately before the end marker is valid: marker is not in prefix.
        item, _ = candidate_at(record(1, [1, 2, 190]), self.tok, 2, 0)
        self.assertEqual(item["cut"], 2)

    def test_intersection_and_sorted_limit_use_same_questions(self):
        rows = [record(8, [1, 2, 3, 2]), record(1, [1, 2, 3, 2]),
                record(4, [1, 2, 190]), record(6, [1, 2, 3, 2])]
        plan = select_common(rows, self.tok, [2, 4], 0, 2)
        self.assertEqual(plan["common_eligible_cases"], [1, 6, 8])
        self.assertEqual(plan["cohort_cases"], [1, 6])
        self.assertEqual([(x["case"], x["cut"]) for x in plan["selected"]], [(1, 2), (1, 4), (6, 2), (6, 4)])
        self.assertEqual(plan["eligible_by_target"]["2"], [1, 4, 6, 8])

    def test_answer_labels_do_not_affect_selection_or_leak(self):
        rows = [record(1, [1, 2, 3, 2]), record(6, [1, 2, 3, 2])]
        first = select_common(rows, self.tok, [2, 4], 0, 2)
        changed = copy.deepcopy(rows)
        for r in changed:
            r["reference_answer"] = "DIFFERENT_ANSWER"
            r["pred_boxed"] = "DIFFERENT_PREDICTION"
        second = select_common(changed, self.tok, [2, 4], 0, 2)
        self.assertEqual(first, second)
        self.assertNotIn("GOLD_SECRET", json.dumps(first))
        self.assertNotIn("reference_answer", json.dumps(first))

    def test_insufficient_cohort_does_not_silently_shrink(self):
        with self.assertRaisesRegex(ValueError, "Only 1 common"):
            select_common([record(1, [1, 2, 3, 2])], self.tok, [2, 4], 0, 2)
        with self.assertRaisesRegex(ValueError, "Only 0 common"):
            select_common([record(1, [1, 2, 190])], self.tok, [2, 4], 0, 1)

    def test_duplicate_identity_and_overlapping_windows_rejected(self):
        rows = [record(1, [1, 2]), record(2, [1, 2])]
        rows[1]["unique_id"] = rows[0]["unique_id"]
        with self.assertRaisesRegex(ValueError, "Duplicate question"):
            select_common(rows, self.tok, [2], 0, 1)
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_windows([2, 4], 2, 1)

    def test_token_hash_and_target_parsing(self):
        self.assertEqual(token_hash([1, 2, 3]), hashlib.sha256(b"[1,2,3]").hexdigest())
        self.assertEqual(parse_targets("4096,1024,2048"), [1024, 2048, 4096])
        for value in ["0,2", "2,2", "abc", ""]:
            with self.subTest(value=value), self.assertRaises(Exception):
                parse_targets(value)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Generation errors must not hide unrelated NNSmith internal failures."""
import unittest
from unittest.mock import patch

from autofusedebugger.nnsmith_integration import fuzz_loop as fuzz


class GenerationFailureTest(unittest.TestCase):
    def run_case(self):
        return fuzz.run_single_case(factory=None, opset=[], model_cls=None, case_seed=123)

    def test_missing_solver_model_is_generation_error(self):
        error = fuzz.InternalError("Failed asertion :: Run check_sat first! | expr is None")
        with patch.object(fuzz, "generate_case", side_effect=error):
            result = self.run_case()
        self.assertEqual(result.status, "gen_error")
        self.assertEqual(result.seed, 123)
        self.assertIn("Run check_sat first!", result.error_summary)

    def test_other_internal_errors_propagate(self):
        with patch.object(fuzz, "generate_case", side_effect=fuzz.InternalError("unexpected invariant")):
            with self.assertRaises(fuzz.InternalError):
                self.run_case()

    def test_ordinary_generation_errors_remain_recoverable(self):
        with patch.object(fuzz, "generate_case", side_effect=ValueError("invalid shape")):
            self.assertEqual(self.run_case().status, "gen_error")

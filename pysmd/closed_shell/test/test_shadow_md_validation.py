"""Tests for ShadowMD input validation."""

import unittest
import importlib.util

import numpy as np

from pysmd.closed_shell.shadow_md import ShadowMD


class TestShadowMDValidation(unittest.TestCase):

    def setUp(self):
        self.md = ShadowMD.__new__(ShadowMD)
        self.md._md_timestep = 1.0


    def test_num_tsteps_accepts_python_integer(self):
        self.md.md_num_tsteps = 4

        self.assertEqual(self.md.md_num_tsteps, 4)


    def test_num_tsteps_rejects_non_python_integers(self):
        invalid_values = [True, 4.0, np.int64(4), "4", 0, -1]
        if importlib.util.find_spec("torch"):
            import torch

            invalid_values.append(torch.tensor(4))

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.md.md_num_tsteps = value


if __name__ == "__main__":
    unittest.main()

"""Tests for linear-algebra backend configuration."""

import importlib.util
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from pysmd.lib import linalg_helper


class TestLinalgHelper(unittest.TestCase):

    def setUp(self):
        linalg_helper.set_linalg_backend("numpy")


    def tearDown(self):
        linalg_helper.set_linalg_backend("numpy")


    def test_numpy_is_default_backend(self):
        la = linalg_helper.get_linalg_backend()
        array = np.array([[1.0, 2.0], [3.0, 4.0]])

        np.testing.assert_allclose(la.matmul(array, array), np.matmul(array, array))
        np.testing.assert_allclose(
            la.einsum("ij,ji->", array, array),
            np.einsum("ij,ji->", array, array),
        )
        self.assertEqual(la.linalg.norm(array), np.linalg.norm(array))


    def test_numpy_can_be_selected_explicitly(self):
        linalg_helper.set_linalg_backend("numpy")

        result = linalg_helper.get_linalg_backend().array([1.0, 2.0])
        self.assertIsInstance(result, np.ndarray)


    def test_removed_compatibility_attributes_are_not_exposed(self):
        la = linalg_helper.get_linalg_backend()

        for name in ("dot", "array2string", "integer"):
            with self.subTest(name=name):
                with self.assertRaises(AttributeError):
                    getattr(la, name)


    def test_numpy_conversion_preserves_array_metadata(self):
        class TaggedArray(np.ndarray):
            pass

        tagged = np.ones((2, 2)).view(TaggedArray)
        tagged.ecoul = 1.25

        converted = linalg_helper.get_linalg_backend().asarray(tagged)
        self.assertEqual(converted.ecoul, 1.25)


    def test_backend_must_be_configured(self):
        with mock.patch.object(linalg_helper, "_backend", None):
            with self.assertRaisesRegex(RuntimeError, "has not been configured"):
                linalg_helper.get_linalg_backend()


    def test_unsupported_backend_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported linear-algebra backend"):
            linalg_helper.set_linalg_backend("unsupported")


    def test_existing_proxy_observes_backend_change(self):
        la = linalg_helper.get_linalg_backend()
        sentinel = object()
        backend = SimpleNamespace(marker=sentinel)

        with mock.patch.dict(linalg_helper._BACKENDS, {"test": backend}):
            linalg_helper.set_linalg_backend("test")
            self.assertIs(la.marker, sentinel)


    def test_failed_torch_import_preserves_backend(self):
        numpy_backend = linalg_helper._backend

        with mock.patch.dict(linalg_helper._BACKENDS, {"torch": None}):
            with mock.patch.dict(sys.modules, {"torch": None}):
                with self.assertRaisesRegex(ImportError, "torch is not installed"):
                    linalg_helper.set_linalg_backend("torch")

        self.assertIs(linalg_helper._backend, numpy_backend)

    def test_failed_cupy_import_preserves_backend(self):
        numpy_backend = linalg_helper._backend
        numpy_name = linalg_helper.get_linalg_backend_name()

        with mock.patch.dict(linalg_helper._BACKENDS, {"cupy": None}):
            with mock.patch.dict(sys.modules, {"cupy": None}):
                with self.assertRaisesRegex(ImportError, "cupy is not importable"):
                    linalg_helper.set_linalg_backend("cupy")

        self.assertIs(linalg_helper._backend, numpy_backend)
        self.assertEqual(linalg_helper.get_linalg_backend_name(), numpy_name)



@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TestTorchBackend(unittest.TestCase):

    def setUp(self):
        linalg_helper.set_linalg_backend("torch")
        self.la = linalg_helper.get_linalg_backend()


    def tearDown(self):
        linalg_helper.set_linalg_backend("numpy")


    def test_torch_operations_match_numpy(self):
        import torch

        left = np.array([[1.0, 2.0], [3.0, 4.0]])
        right = np.array([[2.0, 0.0], [1.0, 2.0]])
        tensor = self.la.asarray(left)

        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.device.type, "cpu")
        self.assertEqual(tensor.dtype, torch.float64)
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.matmul(left, right)),
            np.matmul(left, right),
        )
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.matmul(self.la.matmul(left, right), left)),
            np.matmul(np.matmul(left, right), left),
        )
        np.testing.assert_allclose(
            self.la.to_numpy(
                self.la.einsum("ipq,jpq->ij", left[None], right[None], optimize=True)
            ),
            np.einsum("ipq,jpq->ij", left[None], right[None], optimize=True),
        )


    def test_torch_linalg_and_array_operations_match_numpy(self):
        matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
        values, vectors = self.la.linalg.eigh(matrix)

        expected_values, expected_vectors = np.linalg.eigh(matrix)
        np.testing.assert_allclose(self.la.to_numpy(values), expected_values)
        np.testing.assert_allclose(
            np.abs(self.la.to_numpy(vectors)), np.abs(expected_vectors)
        )
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.linalg.inv(matrix)), np.linalg.inv(matrix)
        )
        self.assertAlmostEqual(self.la.linalg.norm(matrix), np.linalg.norm(matrix))
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.expit(np.array([-1.0, 0.0, 1.0]))),
            np.array([0.2689414213699951, 0.5, 0.7310585786300049]),
        )
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.stack((matrix, matrix))),
            np.stack((matrix, matrix)),
        )

        copied = self.la.copy(self.la.asarray(matrix))
        copied[0, 0] = 9.0
        self.assertEqual(matrix[0, 0], 2.0)

        occupations = self.la.asarray([1.0, 0.5, 0.0])
        formatted = (
            f"[{' '.join(format(x, '.2f') for x in (2.0 * occupations).tolist())}]"
        )
        self.assertEqual(formatted, "[2.00 1.00 0.00]")


    def test_torch_preserves_integer_and_boolean_dtypes(self):
        import torch

        integers = self.la.asarray(np.array([1, 2], dtype=np.int64))
        booleans = self.la.asarray(np.array([True, False]))

        self.assertEqual(integers.dtype, torch.int64)
        self.assertEqual(booleans.dtype, torch.bool)


    def test_torch_conversion_preserves_array_metadata(self):
        class TaggedArray(np.ndarray):
            pass

        tagged = np.ones((2, 2)).view(TaggedArray)
        tagged.ecoul = 1.25
        tagged.exc = -0.5

        tensor = self.la.asarray(tagged)
        converted = self.la.to_numpy(tensor)

        self.assertEqual(tensor.ecoul, 1.25)
        self.assertEqual(converted.ecoul, 1.25)
        self.assertEqual(converted.exc, -0.5)


@unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
class TestCupyBackend(unittest.TestCase):

    def setUp(self):
        import cupy

        try:
            if cupy.cuda.runtime.getDeviceCount() < 1:
                self.skipTest("No CUDA device is available")
        except cupy.cuda.runtime.CUDARuntimeError as exc:
            self.skipTest(f"CUDA is unavailable: {exc}")
        linalg_helper.set_linalg_backend("cupy")
        self.la = linalg_helper.get_linalg_backend()


    def tearDown(self):
        linalg_helper.set_linalg_backend("numpy")


    def test_operations_match_numpy(self):
        import cupy

        matrix = np.array([[2.0, 1.0], [1.0, 3.0]])
        right = np.array([[1.0, 2.0], [3.0, 1.0]])
        device = self.la.asarray(matrix.astype(np.float32))

        self.assertIsInstance(device, cupy.ndarray)
        self.assertEqual(device.dtype, cupy.float64)
        self.assertEqual(linalg_helper.get_linalg_backend_name(), "cupy")
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.matmul(matrix, right)),
            np.matmul(matrix, right),
        )
        np.testing.assert_allclose(
            self.la.to_numpy(
                self.la.einsum("ij,ji->", matrix, right, optimize=True)
            ),
            np.einsum("ij,ji->", matrix, right, optimize=True),
        )
        values, vectors = self.la.linalg.eigh(matrix)
        expected_values, expected_vectors = np.linalg.eigh(matrix)
        np.testing.assert_allclose(self.la.to_numpy(values), expected_values)
        np.testing.assert_allclose(
            np.abs(self.la.to_numpy(vectors)), np.abs(expected_vectors)
        )
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.linalg.inv(matrix)), np.linalg.inv(matrix)
        )
        self.assertAlmostEqual(self.la.linalg.norm(matrix), np.linalg.norm(matrix))
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.expit([-1.0, 0.0, 1.0])),
            [0.2689414213699951, 0.5, 0.7310585786300049],
        )
        np.testing.assert_allclose(
            self.la.to_numpy(self.la.stack((matrix, matrix))),
            np.stack((matrix, matrix)),
        )


    def test_dtype_and_metadata_round_trip(self):
        import cupy

        class TaggedArray(np.ndarray):
            pass

        tagged = np.ones((2, 2), dtype=np.float32).view(TaggedArray)
        tagged.ecoul = 1.25
        tagged.mo_coeff = np.eye(2)

        device = self.la.asarray(tagged)
        converted = self.la.to_numpy(device)
        integers = self.la.asarray(np.array([1, 2], dtype=np.int32))
        booleans = self.la.asarray(np.array([True, False]))
        complex_values = self.la.asarray(np.array([1j], dtype=np.complex64))

        self.assertEqual(device.dtype, cupy.float64)
        self.assertEqual(integers.dtype, cupy.int32)
        self.assertEqual(booleans.dtype, cupy.bool_)
        self.assertEqual(complex_values.dtype, cupy.complex128)
        self.assertEqual(device.ecoul, 1.25)
        self.assertIsInstance(device.mo_coeff, cupy.ndarray)
        self.assertEqual(converted.ecoul, 1.25)
        self.assertIsInstance(converted.mo_coeff, np.ndarray)




if __name__ == "__main__":
    unittest.main()

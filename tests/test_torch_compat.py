import concurrent.futures
import contextlib
import inspect
import io
import itertools
import pickle
import tempfile
import threading
import typing
import unittest
from functools import partial
from pathlib import Path
from typing import ClassVar, Final, Optional, Sequence, Tuple

import torch
import transformers

import tensorizer
import tensorizer.torch_compat as torch_compat
from tensorizer.serialization import DecryptionParams, EncryptionParams
from tensorizer.torch_compat import tensorizer_loading, tensorizer_saving

_ORIG_TORCH_SAVE: Final[callable] = torch.save
_ORIG_TORCH_LOAD: Final[callable] = torch.load

fastest_device: Final[torch.device] = (
    torch.device("cuda", 0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)


class TestTorchCompat(unittest.TestCase):
    MODEL_REF: ClassVar[str] = "EleutherAI/gpt-neo-125M"
    model: ClassVar[torch.nn.Module]
    orig_tensors: ClassVar[Sequence[Tuple[str, torch.Tensor]]]
    tmp_dir: ClassVar[tempfile.TemporaryDirectory]
    tmp_dir_path: ClassVar[Path]
    pt_path: ClassVar[Path]
    tensors_path: ClassVar[Path]
    reference_save_path: ClassVar[Path]
    reference_save_size: ClassVar[int]

    @classmethod
    def load_reference_model(cls, dtype, device=torch.device("cpu")):
        with device:
            return transformers.AutoModelForCausalLM.from_pretrained(
                cls.MODEL_REF
            ).to(dtype)

    @classmethod
    def setUpClass(cls):
        cls.model = cls.load_reference_model(dtype=torch.float16)
        cls.addClassCleanup(delattr, cls, "model")
        cls.model.eval()
        cls.orig_tensors: Sequence[Tuple[str, torch.Tensor]] = (
            cls.extract_tensors(cls.model)
        )
        cls.addClassCleanup(delattr, cls, "orig_tensors")

        cls.tmp_dir = tempfile.TemporaryDirectory(prefix="test_torch_compat")
        cls.tmp_dir.__enter__()
        cls.addClassCleanup(cls.tmp_dir.__exit__, None, None, None)

        cls.tmp_dir_path = Path(cls.tmp_dir.name)
        cls.pt_path = cls.tmp_dir_path / "test.pt"
        cls.tensors_path = cls.tmp_dir_path / "test.pt.tensors"

        # For use as a reference
        cls.reference_save_path = cls.tmp_dir_path / "reference.pt"
        torch.save(cls.model, cls.reference_save_path)
        cls.reference_save_size = cls.reference_save_path.stat().st_size

    @staticmethod
    def extract_tensors(
        model: torch.nn.Module,
    ) -> Sequence[Tuple[str, torch.Tensor]]:
        return list(model.state_dict().items())

    def setUp(self):
        self.assertFalse(self.pt_path.exists())
        self.assertFalse(self.tensors_path.exists())

    def tearDown(self):
        self.pt_path.unlink(missing_ok=True)
        self.tensors_path.unlink(missing_ok=True)

    def check_model(
        self,
        loaded_model: torch.nn.Module,
        expect_device: Optional[torch.device] = None,
        reference: Optional[Sequence[Tuple[str, torch.Tensor]]] = None,
    ):
        loaded_model.eval()
        loaded_tensors = self.extract_tensors(loaded_model)
        orig_tensors = self.orig_tensors if reference is None else reference
        orig_keys = [k for k, _ in orig_tensors]
        loaded_keys = [k for k, _ in loaded_tensors]
        self.assertListEqual(orig_keys, loaded_keys)
        has_tensors: bool = False
        _i = 0
        for (name, tensor), (loaded_name, loaded_tensor) in zip(
            orig_tensors, loaded_tensors
        ):
            has_tensors = True
            _i += 1
            self.assertEqual(name, loaded_name)
            self.assertEqual(tensor.size(), loaded_tensor.size())
            self.assertEqual(tensor.stride(), loaded_tensor.stride())
            self.assertEqual(tensor.dtype, loaded_tensor.dtype)
            if expect_device is not None:
                self.assertEqual(loaded_tensor.device, expect_device)
            if loaded_tensor.device != tensor.device:
                loaded_tensor = loaded_tensor.to(tensor.device)
            self.assertTrue(torch.equal(tensor, loaded_tensor))
        self.assertTrue(has_tensors)

    def check_save_load_signatures(
        self, save_func: callable, load_func: callable
    ):
        # Ensure that the function signatures of torch.save and torch.load
        # match what the wrapper code expects them to be.
        empty = inspect.Parameter.empty
        expected_save_signature = (
            ("obj", empty),
            ("f", empty),
            ("pickle_module", pickle),
        )
        expected_load_signature = (
            ("f", empty),
            ("map_location", None),
            ("pickle_module", torch_compat._LOAD_WRAPPER_DEFAULT_MODULE),
        )
        for func, expected in (
            (save_func, expected_save_signature),
            (load_func, expected_load_signature),
        ):
            params = inspect.signature(
                func, follow_wrapped=False
            ).parameters.values()
            for param, (name, default) in zip(params, expected):
                self.assertEqual(param.name, name)
                self.assertEqual(param.default, default)

    def test_signatures(self):
        with self.subTest("Testing torch signatures"):
            self.assertIs(torch.save, _ORIG_TORCH_SAVE)
            self.assertIs(torch.load, _ORIG_TORCH_LOAD)
            self.check_save_load_signatures(torch.save, torch.load)

        with self.subTest(
            "Testing wrapper signatures"
        ), tensorizer_saving(), tensorizer_loading():
            self.assertIsNot(torch.save, _ORIG_TORCH_SAVE)
            self.assertIsNot(torch.load, _ORIG_TORCH_LOAD)
            self.check_save_load_signatures(torch.save, torch.load)

    def test_torch_load(self):
        # Sanity check
        with torch.device("cpu"):
            self.assertIs(torch.load, _ORIG_TORCH_LOAD)
            loaded_model: torch.nn.Module = torch.load(
                self.reference_save_path, weights_only=False
            )
        self.assertFalse(
            self.reference_save_path.with_suffix(".pt.tensors").exists()
        )
        self.check_model(loaded_model, torch.device("cpu"))

    def test_save_load(self):
        with tensorizer_saving():
            torch.save(self.model, self.pt_path)
        self.assertTrue(self.pt_path.is_file())
        self.assertTrue(self.tensors_path.is_file())

        with self.subTest("Testing file sizes"):
            pt_size: int = self.pt_path.stat().st_size
            tensors_size: int = self.tensors_path.stat().st_size
            self.assertLess(pt_size, self.reference_save_size)
            self.assertLess(pt_size, 1 << 20)  # Should be less than 1 MiB
            self.assertGreater(tensors_size, pt_size)
            self.assertGreater(tensors_size, 20 << 20)  # More than 20 MiB
            self.assertLess(tensors_size, int(self.reference_save_size * 1.8))

        with self.subTest("Testing loading"):
            with tensorizer_loading(device=fastest_device):
                loaded_model = torch.load(self.pt_path, weights_only=False)
            loaded_model.eval()
            self.check_model(loaded_model, torch.device(fastest_device))
            # Check that it can process a forward pass
            loaded_model(torch.tensor((1000,), device=fastest_device))

    # def test_save_load_s3(self):
    #     pass

    def test_save_load_args(self):
        encryption = EncryptionParams.random()
        decryption = DecryptionParams.from_key(encryption.key)
        tensors_path = self.tensors_path.with_suffix(".tensors.test")
        self.addCleanup(tensors_path.unlink, missing_ok=True)
        with tensorizer_saving(tensors_path, encryption=encryption):
            torch.save(self.model, self.pt_path)
        self.assertTrue(self.pt_path.is_file())
        self.assertTrue(tensors_path.is_file())

        with self.subTest("Testing loading"):
            with tensorizer_loading(
                tensors_path, device=fastest_device, encryption=decryption
            ):
                loaded_model = torch.load(self.pt_path, weights_only=False)
            loaded_model.eval()
            self.check_model(loaded_model, fastest_device)
            del loaded_model

        with self.subTest("Testing invalid loading"), self.assertRaises(
            tensorizer.CryptographyError
        ), tensorizer_loading(tensors_path, device=fastest_device):
            torch.load(self.pt_path, weights_only=False)

    def test_save_load_fp8_torch(self):
        dtype = torch.float8_e4m3fn
        model = self.load_reference_model(dtype=dtype, device=fastest_device)
        with tensorizer_saving():
            torch.save(model, self.pt_path)
        self.assertTrue(self.pt_path.is_file())
        self.assertTrue(self.tensors_path.is_file())

        with self.subTest("Testing loading"):
            with tensorizer_loading(device=fastest_device):
                loaded_model = torch.load(self.pt_path, weights_only=False)
            loaded_model.eval()
            dtypes = {
                tensor.dtype for _, tensor in self.extract_tensors(loaded_model)
            }
            self.assertIn(dtype, dtypes)
            self.assertNotIn(torch.float16, dtypes)
            self.check_model(
                loaded_model,
                fastest_device,
                reference=self.extract_tensors(model),
            )

    def test_thread_safety(self):
        start: threading.Barrier = threading.Barrier(parties=2)
        finish: threading.Barrier = threading.Barrier(parties=2)
        model_1: torch.nn.Module = self.model
        model_2: torch.nn.Module = self.load_reference_model(torch.float16)

        def _save_load_tensorizer(
            model: torch.nn.Module,
            pt_path: Path,
            save_kwargs: dict,
            load_kwargs: dict,
        ) -> torch.nn.Module:
            with tensorizer_saving(**save_kwargs):
                start.wait(timeout=10)
                start.reset()
                torch.save(model, pt_path)
                finish.wait(timeout=10)
                finish.reset()

            with tensorizer_loading(**load_kwargs):
                start.wait(timeout=10)
                start.reset()
                try:
                    return torch.load(pt_path)
                finally:
                    finish.wait(timeout=10)
                    finish.reset()

        def _save_load_torch(
            model: torch.nn.Module, pt_path: Path
        ) -> torch.nn.Module:
            start.wait(timeout=10)
            torch.save(model, pt_path)
            finish.wait(timeout=10)

            start.wait(timeout=10)
            try:
                return torch.load(pt_path, weights_only=False)
            finally:
                finish.wait(timeout=10)

        pt_path_1 = self.pt_path
        tensors_path_1 = self.tensors_path
        pt_path_2 = self.tmp_dir_path / "test-2.pt"
        tensors_path_2 = self.tmp_dir_path / "test-2.pt.tensors"
        paths = (pt_path_1, tensors_path_1, pt_path_2, tensors_path_2)
        for path in paths:
            self.addCleanup(path.unlink, missing_ok=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            with self.subTest("One thread active, one thread inactive"):
                f1 = pool.submit(
                    _save_load_tensorizer, model_1, pt_path_1, {}, {}
                )
                f2 = pool.submit(_save_load_torch, model_2, pt_path_2)
                m1 = f1.result()
                m2 = f2.result()
                self.check_model(m1)
                self.check_model(m2)
                self.assertTrue(pt_path_1.is_file())
                self.assertTrue(tensors_path_1.is_file())
                self.assertTrue(pt_path_2.is_file())
                self.assertFalse(tensors_path_2.exists())
            for path in paths:
                path.unlink(missing_ok=True)

            start.reset()
            finish.reset()

            with self.subTest("Two threads with different active contexts"):
                encryption_1 = EncryptionParams.random()
                decryption_1 = DecryptionParams.from_key(encryption_1.key)
                encryption_2 = EncryptionParams.random()
                decryption_2 = DecryptionParams.from_key(encryption_2.key)
                self.assertNotEqual(encryption_1.key, encryption_2.key)
                f1 = pool.submit(
                    _save_load_tensorizer,
                    model_1,
                    pt_path_1,
                    {"encryption": encryption_1},
                    {"encryption": decryption_1},
                )
                f2 = pool.submit(
                    _save_load_tensorizer,
                    model_2,
                    pt_path_2,
                    {"encryption": encryption_2},
                    {"encryption": decryption_2},
                )
                m1 = f1.result()
                m2 = f2.result()
                self.check_model(m1)
                self.check_model(m2)
                for path in paths:
                    self.assertTrue(path.is_file())

    def test_shared_storage(self):
        a = torch.tensor((1, 2, 3), dtype=torch.long)
        b = torch.tensor((4, 5, 6), dtype=torch.long)
        c = torch.tensor((), dtype=torch.long)
        # noinspection PyTypeChecker
        c.set_(a)
        self.assertTrue(c.is_set_to(a))
        self.assertEqual(a.data_ptr(), c.data_ptr())
        self.assertTrue(torch.equal(a, c))
        d = c.view(dtype=torch.float64)
        self.assertTrue(d.is_set_to(a))
        tensors = [a, b, c, d]
        with tensorizer_saving():
            torch.save(tensors, self.pt_path)
        self.assertTrue(self.pt_path.is_file())
        self.assertTrue(self.tensors_path.is_file())

        with tensorizer_loading(device="cpu"):
            _a, _b, _c, _d = torch.load(self.pt_path, weights_only=False)
        for orig, loaded in ((a, _a), (b, _b), (c, _c), (d, _d)):
            self.assertTrue(torch.equal(orig, loaded))
            self.assertEqual(orig.dtype, loaded.dtype)
        self.assertTrue(torch.equal(_a, _c))
        self.assertEqual(_a.data_ptr(), _c.data_ptr())
        self.assertTrue(_c.is_set_to(_a))
        self.assertTrue(_d.is_set_to(_a))

    def test_suppress_weights_only(self):
        tensors = list(torch.arange(16).view((4, 4)))

        with tensorizer_saving():
            torch.save(tensors, self.pt_path)
        with self.assertRaisesRegex(
            RuntimeError,
            "Can not safely load weights"
            " when explicit pickle_module is specified",
        ), tensorizer_loading(device="cpu"):
            torch.load(self.pt_path, weights_only=True)
        with tensorizer_loading(device="cpu", suppress_weights_only=True):
            loaded_tensors = torch.load(self.pt_path, weights_only=True)

        self.assertTrue(
            torch.equal(torch.stack(tensors), torch.stack(loaded_tensors))
        )

    def test_meta_tensors(self):
        t1 = torch.tensor((1, 2, 3, 4), dtype=torch.long)
        t2 = t1.to(device="meta")

        with tensorizer_saving():
            torch.save([t1, t2], self.pt_path)

        with tensorizer_loading(device="cpu"):
            loaded_t1, loaded_t2 = torch.load(self.pt_path)

        self.assertTrue(torch.equal(t1, loaded_t1))
        self.assertTrue(loaded_t2.is_meta)
        self.assertEqual(t2.dtype, loaded_t2.dtype)
        self.assertEqual(t2.size(), loaded_t2.size())
        self.assertEqual(t2.stride(), loaded_t2.stride())

    def test_name_callback(self):
        dynamic_tensors_path: Path = self.pt_path.with_suffix(
            ".pt.tensors.dynamic"
        )

        def path_callback(f: torch.types.FileLike) -> io.BufferedIOBase:
            # Test with an exotic function that returns a pre-opened
            # stream dynamically, based on the input file's name
            _path = Path(f).with_suffix(".pt.tensors.dynamic")
            if not _path.exists():
                self.addCleanup(_path.unlink, missing_ok=True)
            file_obj = _path.open("rb" if _path.exists() else "wb+")
            self.addCleanup(file_obj.close)
            return typing.cast(io.BufferedIOBase, file_obj)

        with tensorizer_saving(path_callback):
            torch.save(self.model, self.pt_path)

        self.assertTrue(self.pt_path.is_file())
        self.assertFalse(self.tensors_path.exists())
        self.assertTrue(dynamic_tensors_path.is_file())

        with tensorizer_loading(path_callback, device="cpu"):
            loaded_model = torch.load(self.pt_path)

        self.check_model(loaded_model)

    def test_nested_contexts(self):
        sd = {
            f"layer.{i:d}": torch.randn(
                (16, 16), device="cpu", dtype=torch.float32
            )
            for i in range(4)
        }
        keys = tuple(sd.keys())

        def check_sd(_sd):
            self.assertTupleEqual(keys, tuple(_sd.keys()))
            for name, tensor in sd.items():
                self.assertTrue(torch.equal(tensor, _sd[name]))

        # These produce reusable callables with frozen args
        cpu_loading = partial(tensorizer_loading, device="cpu")
        saving = partial(partial, tensorizer_saving)
        loading = partial(partial, cpu_loading)

        def permuted(context1, context2):
            @contextlib.contextmanager
            def _ctx():
                with self.subTest(f"{name1} + {name2}"), ctx1(), ctx2():
                    yield

            for (name1, ctx1), (name2, ctx2) in itertools.permutations(
                (context1, context2)
            ):
                yield _ctx

        for ctx in permuted(
            ("tensorizer_saving", saving()),
            ("tensorizer_loading", loading()),
        ):
            with ctx():
                torch.save(sd, self.pt_path)
                self.assertTrue(self.pt_path.is_file())
                self.assertTrue(self.tensors_path.is_file())
                check_sd(torch.load(self.pt_path))
            self.pt_path.unlink(missing_ok=True)
            self.tensors_path.unlink(missing_ok=True)

        alt_tensors_path = self.tmp_dir_path / "test-2.pt.tensors"
        self.addCleanup(alt_tensors_path.unlink, missing_ok=True)

        def cleanup() -> None:
            self.pt_path.unlink(missing_ok=True)
            self.tensors_path.unlink(missing_ok=True)
            alt_tensors_path.unlink(missing_ok=True)

        def check_saved_primary() -> None:
            self.assertTrue(self.pt_path.is_file())
            self.assertTrue(self.tensors_path.is_file())
            self.assertFalse(alt_tensors_path.exists())

        def check_saved_alt() -> None:
            self.assertTrue(self.pt_path.is_file())
            self.assertFalse(self.tensors_path.exists())
            self.assertTrue(alt_tensors_path.is_file())

        #
        # Test mixing tensorizer_saving and tensorizer_loading together
        #

        # Try saving to an alternate path but not loading from it
        for ctx in permuted(
            ("tensorizer_saving(path)", saving(alt_tensors_path)),
            ("tensorizer_loading", loading()),
        ):
            with ctx():
                torch.save(sd, self.pt_path)
                check_saved_alt()
                with self.assertRaises(OSError):
                    torch.load(self.pt_path)
            cleanup()

        # Try loading from an alternate path but not saving to it
        for ctx in permuted(
            ("tensorizer_saving", saving()),
            ("tensorizer_loading(path)", loading(alt_tensors_path)),
        ):
            with ctx():
                torch.save(sd, self.pt_path)
                check_saved_primary()
                with self.assertRaises(OSError):
                    torch.load(self.pt_path)
            cleanup()

        # Try both saving to and loading from an alternate path
        for ctx in permuted(
            ("tensorizer_saving(path)", saving(alt_tensors_path)),
            ("tensorizer_loading(path)", loading(alt_tensors_path)),
        ):
            with ctx():
                torch.save(sd, self.pt_path)
                check_saved_alt()
                check_sd(torch.load(self.pt_path))
            cleanup()

        #
        # Test nesting multiple levels of the same type of context manager
        # The most recent context should take precedence
        #

        # Nested saving context managers
        for save_name, default_save in (
            ("tensorizer_saving", saving()),
            ("tensorizer_saving(default)", saving(self.tensors_path)),
        ):
            with self.subTest(f"{save_name} + tensorizer_saving(path)"):
                with default_save(), tensorizer_saving(alt_tensors_path):
                    torch.save(sd, self.pt_path)
                check_saved_alt()
                with cpu_loading(alt_tensors_path):
                    check_sd(torch.load(self.pt_path))
            cleanup()

            with self.subTest(f"tensorizer_saving(path) + {save_name}"):
                with tensorizer_saving(alt_tensors_path), default_save():
                    torch.save(sd, self.pt_path)
                check_saved_primary()
                with cpu_loading():
                    check_sd(torch.load(self.pt_path))
            cleanup()

            # Make sure an outer context is restored
            # correctly after leaving an inner context
            with self.subTest(f"tensorizer_saving(path) after {save_name}"):
                with tensorizer_saving(alt_tensors_path):
                    with default_save():
                        # This should temporarily change the context,
                        # but bring it back once the block is over.
                        pass
                    torch.save(sd, self.pt_path)
                check_saved_alt()
                with cpu_loading(alt_tensors_path):
                    check_sd(torch.load(self.pt_path))
            cleanup()

            with self.subTest(f"{save_name} after tensorizer_saving(path)"):
                with default_save():
                    with tensorizer_saving(alt_tensors_path):
                        # This should temporarily change the context,
                        # but bring it back once the block is over.
                        pass
                    torch.save(sd, self.pt_path)
                check_saved_primary()
                with cpu_loading():
                    check_sd(torch.load(self.pt_path))
            cleanup()

        # Nested loading context managers
        for load_name, default_load in (
            ("tensorizer_loading", loading()),
            ("tensorizer_loading(default)", loading(self.tensors_path)),
        ):
            with self.subTest(f"{load_name} + tensorizer_loading(path)"):
                with tensorizer_saving(alt_tensors_path):
                    torch.save(sd, self.pt_path)
                check_saved_alt()
                with default_load(), cpu_loading(alt_tensors_path):
                    check_sd(torch.load(self.pt_path))
            cleanup()

            with self.subTest(f"tensorizer_loading(path) + {load_name}"):
                with tensorizer_saving():
                    torch.save(sd, self.pt_path)
                check_saved_primary()
                with cpu_loading(alt_tensors_path), default_load():
                    check_sd(torch.load(self.pt_path))
            cleanup()

            # Make sure an outer context is restored
            # correctly after leaving an inner context
            with self.subTest(f"tensorizer_loading(path) after {save_name}"):
                with tensorizer_saving(alt_tensors_path):
                    torch.save(sd, self.pt_path)
                check_saved_alt()
                with cpu_loading(alt_tensors_path):
                    with default_load():
                        # This should temporarily change the context,
                        # but bring it back once the block is over.
                        pass
                    check_sd(torch.load(self.pt_path))
            cleanup()

            with self.subTest(f"{save_name} after tensorizer_loading(path)"):
                with tensorizer_saving():
                    torch.save(sd, self.pt_path)
                check_saved_primary()
                with cpu_loading():
                    with cpu_loading(alt_tensors_path):
                        # This should temporarily change the context,
                        # but bring it back once the block is over.
                        pass
                    check_sd(torch.load(self.pt_path))
            cleanup()

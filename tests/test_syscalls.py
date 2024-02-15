import os
import tempfile
import unittest

import tensorizer._syscalls as syscalls


class TestSyscalls(unittest.TestCase):
    def test_fallocate(self):
        from tensorizer._syscalls import try_fallocate

        has_fallocate: bool = syscalls.has_fallocate()

        with tempfile.NamedTemporaryFile(mode="wb+") as file:
            fd: int = file.fileno()
            with self.subTest("Regular fallocate"):
                self.assertEqual(
                    try_fallocate(fd=fd, offset=50, length=1000),
                    has_fallocate,
                )
                try:
                    self.assertEqual(
                        os.stat(fd).st_size, 1050 if has_fallocate else 0
                    )
                finally:
                    os.ftruncate(fd, 0)
            if not has_fallocate:
                # The rest of the tests check for errors, which cannot be raised
                # if the fallocate syscall is not actually available.
                return
            with self.subTest(
                "Invalid fallocate invocation, errors suppressed"
            ):
                self.assertFalse(
                    try_fallocate(
                        fd=fd, offset=-1, length=0, suppress_all_errors=True
                    )
                )
            with self.subTest(
                "Invalid fallocate invocation (bad offset and length)"
            ), self.assertRaises(OSError):
                try_fallocate(fd=fd, offset=-1, length=0)
            self.assertEqual(os.stat(fd).st_size, 0)
        with self.subTest(
            "Invalid fallocate invocation (bad file descriptor)"
        ), self.assertRaises(OSError):
            try:
                r_fd, w_fd = os.pipe()
                try_fallocate(fd=w_fd, offset=0, length=1000)
            finally:
                os.close(r_fd)
                os.close(w_fd)

    @unittest.skipUnless(
        hasattr(os, "pwrite"), "pwrite must be available to test pwrite"
    )
    def test_out_of_order_pwrite(self):
        def _filler(length: int) -> bytes:
            mul, rem = divmod(length, 10)
            return (b"0123456789" * (mul + (rem != 0)))[:length]

        with tempfile.TemporaryFile("wb+", buffering=0) as file:
            fd: int = file.fileno()

            def pwrite(buffer: bytes, offset: int) -> None:
                self.assertEqual(os.pwrite(fd, buffer, offset), len(buffer))

            discontiguous_offset: int = (10 << 10) + 5
            end_contents: bytes = _filler(10)
            expected_size: int = discontiguous_offset + len(end_contents)
            # This should fill the file with zeroes up to discontiguous_offset
            pwrite(end_contents, discontiguous_offset)
            self.assertEqual(os.stat(fd).st_size, expected_size)
            start_contents: bytes = _filler(5 << 10)
            # This should overwrite the existing zeroes,
            # and not change the length of the file
            pwrite(start_contents, 0)
            self.assertEqual(os.stat(fd).st_size, expected_size)
            total_written: int = len(start_contents) + len(end_contents)
            # The expected end result is start_contents,
            # a gap of zeroes up to discontiguous_offset, and then end_contents
            expected_contents: bytes = (
                start_contents
                + bytes(expected_size - total_written)
                + end_contents
            )
            self.assertEqual(file.read(), expected_contents)

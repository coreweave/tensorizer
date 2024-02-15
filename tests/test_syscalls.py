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

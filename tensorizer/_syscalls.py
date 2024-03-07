import ctypes
import errno
import mmap

__all__ = (
    "has_fallocate",
    "try_fallocate",
    "prefault",
)


try:
    _libc = ctypes.CDLL(None)
except TypeError:
    _libc = ctypes.pythonapi

_IN: int = 1


def _errcheck(result, func, args) -> None:
    del args
    if result == -1:
        err: int = ctypes.get_errno()
        str_err: str = errno.errorcode.get(err, "Unknown error")
        raise OSError(err, str_err)
    elif result == 0:
        return None
    else:
        raise OSError("Unknown return code")


def _get_fallocate():
    from ctypes import CFUNCTYPE, c_int, c_longlong

    prototype = CFUNCTYPE(
        c_int,
        c_int,
        c_int,
        c_longlong,
        c_longlong,
        use_errno=True,
    )
    paramflags = (
        (_IN, "fd"),
        (_IN, "mode"),
        (_IN, "offset"),
        (_IN, "len"),
    )

    try:
        _func = prototype(("fallocate", _libc), paramflags)
    except AttributeError:
        return None
    _func.errcheck = _errcheck

    return _func


_fallocate = _get_fallocate()
del _get_fallocate


def has_fallocate() -> bool:
    """
    Checks if the Linux ``fallocate(2)`` syscall is available.
    Returns: ``True`` if ``fallocate(2)`` is available, ``False`` otherwise.
    """
    return _fallocate is not None


def try_fallocate(
    fd: int, offset: int, length: int, suppress_all_errors: bool = False
) -> bool:
    """
    Calls ``fallocate(2)`` on the given file descriptor `fd` if available,
    ignoring some errors if unsuccessful.

    Args:
        fd: File descriptor on which to call ``fallocate(2)``.
        offset: Starting position of the byte range to allocate.
        length: Number of bytes to allocate.
        suppress_all_errors: If True, ignore all errors from unsuccessful calls.
            Otherwise, only ignores ``EOPNOTSUPP``.

    Returns: ``True`` if fallocate ran successfully, ``False`` otherwise.
    Raises:
        OSError: If `suppress_all_errors` is ``False`` and the call failed
            due to an error other than ``EOPNOTSUPP``.
    """
    if _fallocate is None:
        return False
    try:
        _fallocate(fd=fd, mode=0, offset=offset, len=length)
        return True
    except OSError as e:
        if suppress_all_errors or e.errno == errno.EOPNOTSUPP:
            return False
        else:
            raise


def _get_madvise():
    from ctypes import CFUNCTYPE, c_int, c_size_t, c_void_p

    prototype = CFUNCTYPE(
        c_int,
        c_void_p,
        c_size_t,
        c_int,
        use_errno=True,
    )
    paramflags = (
        (_IN, "addr"),
        (_IN, "length"),
        (_IN, "advice"),
    )

    try:
        _func = prototype(("madvise", _libc), paramflags)
    except AttributeError:
        return None
    _func.errcheck = _errcheck

    return _func


_madvise = _get_madvise()
del _get_madvise

_madv_populate_write: int = 23


def _can_prefault_with_madvise() -> bool:
    if _madvise is None or _libc is ctypes.pythonapi:
        # If _libc is ctypes.pythonapi then the call would hold the GIL
        return False
    n: int = mmap.PAGESIZE
    private: int = getattr(mmap, "MAP_PRIVATE", 0)
    flags = {} if private == 0 else {"flags": private}
    with mmap.mmap(-1, n, **flags) as m:
        try:
            # MADV_POPULATE_WRITE is only available on Linux 5.14 and up
            _madvise(
                ctypes.byref((ctypes.c_ubyte * n).from_buffer(m)),
                n,
                _madv_populate_write,
            )
        except OSError:
            return False
        else:
            return True


if _can_prefault_with_madvise():

    def prefault(address, length: int):
        _madvise(address, length, _madv_populate_write)

else:

    def prefault(address, length: int):
        ctypes.memset(address, 0x00, length)


del _can_prefault_with_madvise

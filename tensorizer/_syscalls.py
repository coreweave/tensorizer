import ctypes
import errno

__all__ = (
    "has_fallocate",
    "try_fallocate",
)


try:
    _libc = ctypes.CDLL(None)
except TypeError:
    _libc = ctypes.pythonapi

_IN: int = 1

memcpy = _libc.memcpy
memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
memcpy.restype = ctypes.c_void_p

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
    print('fallocate', offset, length)
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

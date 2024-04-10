"""Monkeypatches Thread.start so that pthreads have a useful name for debugging"""

import ctypes
import threading

__all__ = []

libpthread = ctypes.CDLL(ctypes.util.find_library("pthread"))
pthread_setname_np = libpthread.pthread_setname_np
pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
pthread_setname_np.restype = ctypes.c_int


orig_start = threading.Thread.start


def new_thread_start(self):
    orig_start(self)
    name = self.name
    if name.startswith("Thread-"):
        name = name[len("Thread-") :]
    name = name[-15:].encode()
    ident = self.ident
    pthread_setname_np(ident, name)


threading.Thread.start = new_thread_start

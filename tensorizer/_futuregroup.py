import concurrent.futures
from typing import Any, List, Optional, Sequence, Union


class _FutureGroup:
    def __init__(self, futures):  # type: (Sequence[_Future]) -> None
        self.futures = futures

    def cancel(self) -> bool:
        result = True
        for f in self.futures:
            result = result and f.cancel()
        return result

    def cancelled(self) -> bool:
        return all(f.cancelled() for f in self.futures)

    def running(self) -> bool:
        return any(f.running() for f in self.futures)

    def done(self) -> bool:
        return all(f.done() for f in self.futures)

    def result(self, timeout=None) -> None:
        _future_wait_and_raise(self.futures, timeout=timeout)

    def exception(self, timeout=None) -> Optional[BaseException]:
        for f in self.futures:
            exc = f.exception(timeout=timeout)
            if exc:
                return exc
        return None


_Future = Union[_FutureGroup, concurrent.futures.Future]


def _future_wait_and_raise(futures: Sequence[_Future], timeout=None) -> None:
    # Wait on a list of futures with a timeout. Raise any exceptions, including TimeoutErrors.

    flattened_futures: List[concurrent.futures.Future] = []
    futures = list(futures)
    while futures:
        f = futures.pop()
        if isinstance(f, _FutureGroup):
            futures.extend(f.futures)
        else:
            flattened_futures.append(f)

    fs = concurrent.futures.wait(flattened_futures, timeout=timeout)
    for f in fs.done:
        # if the future has an exception, this will raise it
        f.result()
    for f in fs.not_done:
        # force raise of TimeoutError
        f.result(0)

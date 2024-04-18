import concurrent.futures
from collections.abc import Callable
from typing import Any, Optional, Sequence


class _FutureGroup(concurrent.futures.Future):
    def __init__(self, futures: Sequence[concurrent.futures.Future]):
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

    def result(self, timeout=None) -> Sequence[Any]:
        return _future_wait_and_raise(self.futures, timeout=timeout)

    def exception(self, timeout=None) -> Optional[BaseException]:
        for f in self.futures:
            exc = f.exception(timeout=timeout)
            if exc:
                return exc
        return None

    def add_done_callback(self, _):
        raise NotImplementedError()

    def set_running_or_notify_cancel(self) -> bool:
        raise NotImplementedError()

    def set_result(self, _) -> None:
        raise NotImplementedError()

    def set_exception(self, _) -> None:
        raise NotImplementedError()


def _future_wait_and_raise(
    futures: Sequence[concurrent.futures.Future], timeout=None
) -> Sequence[Any]:
    # Wait on a list of futures with a timeout. Raise any exceptions, including TimeoutErrors.
    # otherwise return the list of results in the same order as the input futures.
    results = []
    fs = concurrent.futures.wait(futures, timeout=timeout)
    for f in fs.done:
        # if the future has an exception, this will raise it
        results.append(f.result())
    for f in fs.not_done:
        # force raise of TimeoutError
        results.append(f.result(0))
    return results

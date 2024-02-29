import enum
import os
import pathlib
import sys
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Union

__all__ = ("effective_cpu_count", "RoundingMode")


class RoundingMode(enum.Enum):
    UP = 1
    DOWN = 2
    HALF_EVEN = 3


if sys.platform == "linux":

    @lru_cache(maxsize=None)
    def _cpu_quota() -> Optional[Fraction]:
        cgroup = pathlib.Path("/sys/fs/cgroup")
        cgroup_v1 = cgroup / "cpu,cpuacct"
        cgroup_v2 = cgroup / "user.slice" / "cpu.max"
        try:
            if not cgroup.is_dir():
                return None
            elif cgroup_v1.is_dir():
                quota, period = (
                    (cgroup_v1 / p).read_text()
                    for p in ("cpu.cfs_quota_us", "cpu.cfs_period_us")
                )
            elif cgroup_v2.is_file():
                quota, period = cgroup_v2.read_text().split()
            else:
                raise OSError()

            if quota == "max":
                return None

            q, p = map(int, (quota, period))
            if q > 0 and p > 0:
                return Fraction(q, p)
            else:
                raise ValueError()
        except (OSError, ValueError):
            return None

else:

    def _cpu_quota() -> None:
        return None


def effective_cpu_count(
    rounding: Optional[RoundingMode] = RoundingMode.UP,
) -> Union[int, Fraction]:
    if not isinstance(rounding, (RoundingMode, type(None))):
        raise TypeError("Invalid type for rounding mode")
    quota: Optional[Fraction] = _cpu_quota()
    if quota is None:
        return os.cpu_count()
    if rounding is None:
        return quota
    else:
        if rounding is RoundingMode.UP:
            return quota.__ceil__()
        elif rounding is RoundingMode.DOWN:
            return quota.__floor__()
        elif rounding is RoundingMode.HALF_EVEN:
            return round(quota)
        else:
            raise ValueError("Unknown rounding mode")

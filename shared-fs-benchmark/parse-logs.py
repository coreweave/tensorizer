#!/bin/env python

import argparse
import bisect
import dataclasses
import json
import statistics
import typing
from decimal import Decimal
from typing import List, MutableMapping, Optional


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="parse benchmark logs")
    parser.add_argument(
        "-i", "--infile", type=argparse.FileType("rb"), default="-"
    )
    parser.add_argument(
        "-o", "--outfile", type=argparse.FileType("w"), default="-"
    )
    parser.add_argument("-k", "--keep-timestamps", action="store_true")
    parser.add_argument("-r", "--round", action="store_true")
    return parser.parse_args(argv)


@dataclasses.dataclass
class Times:
    __slots__ = ("times", "durations")
    times: List[Decimal]
    durations: MutableMapping[str, List[Decimal]]

    def __init__(self, duration_keys=("open", "io", "total")):
        self.times = []
        self.durations = {k: [] for k in duration_keys}

    def log_time(self, t: Decimal):
        bisect.insort(self.times, t)

    @property
    def min_time(self) -> Optional[Decimal]:
        return self.times[0] if self.times else None

    @property
    def max_time(self) -> Optional[Decimal]:
        return self.times[-1] if self.times else None

    @property
    def range(self) -> Optional[Decimal]:
        return self.max_time - self.min_time if self.times else None

    def __bool__(self) -> bool:
        return bool(self.times or any(self.durations.values()))


class DecimalSerializer(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


def round_d(d: Decimal, precision: int) -> Decimal:
    return typing.cast(Decimal, round(d, precision))


def main(argv=None) -> None:
    args = parse_args(argv)
    timestamp_to_ms = Decimal("1e-6")
    timings = {"write": Times(), "read": Times()}
    objs = []
    for line in args.infile:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line, parse_float=Decimal)
            if not isinstance(obj, dict):
                raise TypeError("Not an object")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("Invalid JSONL") from e
        for mode, timing in timings.items():
            try:
                timing.log_time(obj[f"{mode}_start_timestamp"])
                timing.log_time(obj[f"{mode}_end_timestamp"])
                for key, l in timing.durations.items():
                    l.append(round_d(obj[f"{mode}_{key}_duration_ms"], 6))
            except KeyError:
                pass
        objs.append(obj)

    results = {}
    if not timings["read"]:
        del timings["read"]

    for mode, timing in timings.items():
        for key, l in timing.durations.items():
            results[f"median_{mode}_{key}_duration_ms"] = statistics.median(l)
            results[f"stddev_{mode}_{key}_duration_ms"] = statistics.stdev(l)
    for mode, timing in timings.items():
        results[f"combined_{mode}_duration_ms"] = timing.range * timestamp_to_ms
    objs.sort(key=lambda o: o["chunk"])
    for obj in objs:
        for mode, timing in timings.items():
            for timestamp in f"{mode}_start_timestamp", f"{mode}_end_timestamp":
                if timestamp not in obj:
                    continue
                elif args.keep_timestamps:
                    obj[timestamp] -= timing.min_time
                    obj[timestamp] *= timestamp_to_ms
                else:
                    del obj[timestamp]
    for obj in (*objs, results):
        if args.round:
            for k, v in obj.items():
                if isinstance(v, Decimal):
                    obj[k] = round_d(v, 3)
        json.dump(obj, args.outfile, indent=None, cls=DecimalSerializer)
        args.outfile.write("\n")


if __name__ == "__main__":
    main()

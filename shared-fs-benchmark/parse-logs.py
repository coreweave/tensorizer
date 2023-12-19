#!/bin/env python

import argparse
import json
import statistics


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


def main(argv=None) -> None:
    args = parse_args(argv)
    times = []
    read_times = []
    durations = {"open": [], "write": [], "total": []}
    read_durations = {"open": [], "io": [], "total": []}
    objs = []
    for line in args.infile:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError("Not an object")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError("Invalid JSONL") from e
        if "read_start_timestamp" in obj:
            modes = (
                ("", times, durations),
                ("read_", read_times, read_durations),
            )
        else:
            modes = (("", times, durations),)
        for mode, time_log, duration_log in modes:
            time_log.append(obj[f"{mode}start_timestamp"] * 1e3)
            time_log.append(obj[f"{mode}end_timestamp"] * 1e3)
            for key, l in duration_log.items():
                l.append(obj[f"{mode}{key}_duration_ms"])
        objs.append(obj)
    results = {}
    for key, l in durations.items():
        results[f"median_{key}_duration_ms"] = statistics.median(l)
        results[f"stddev_{key}_duration_ms"] = statistics.stdev(l)
    if read_times:
        for key, l in read_durations.items():
            results[f"median_read_{key}_duration_ms"] = statistics.median(l)
            results[f"stddev_read_{key}_duration_ms"] = statistics.stdev(l)
    global_start, global_end = min(times), max(times)
    global_read_start, global_read_end = min(read_times), max(read_times)
    results["combined_duration_ms"] = global_end - global_start
    results["combined_read_duration_ms"] = global_read_end - global_read_start
    objs.sort(key=lambda o: o["chunk"])
    for obj in objs:
        for timestamp in "start_timestamp", "end_timestamp":
            if args.keep_timestamps:
                obj[timestamp] = obj[timestamp] * 1e3 - global_start
            else:
                del obj[timestamp]
        for timestamp in "read_start_timestamp", "read_end_timestamp":
            if timestamp not in obj:
                continue
            if args.keep_timestamps:
                obj[timestamp] = obj[timestamp] * 1e3 - global_read_start
            else:
                del obj[timestamp]
    for obj in (*objs, results):
        if args.round:
            for k, v in obj.items():
                if isinstance(v, float):
                    obj[k] = round(v, 3)
        json.dump(obj, args.outfile, indent=None)
        args.outfile.write("\n")


if __name__ == "__main__":
    main()

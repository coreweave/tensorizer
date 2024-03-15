#!/usr/bin/env python
# coding: utf-8

"""
This script creates several visualizations from the JSONL output of
examples/benchmark_buffer_size/benchmark.py. It is not run automatically
through the job defined in benchmark.yaml, but should instead be run
on cumulative logs gathered from all pods once that job has finished.

The functions in this script are meant to be hackable,
to add new visualizations as needed. The code is loosely organized
into cells, so it can be run either all at once or interactively.
"""


# %% Imports


import argparse
import os
from collections import Counter
from decimal import Decimal
from functools import lru_cache, partial
from io import StringIO
from itertools import chain
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% Argument parsing


# Swap this for a list for interactive use
argv = None

parser = argparse.ArgumentParser(
    description="create visualizations from benchmark outputs"
)
parser.add_argument(
    "jsonl_file",
    metavar="JSONL_FILE",
    nargs="+",
    help="output logs from the benchmark.py script",
    type=Path,
)
parser.add_argument(
    "-n",
    "--node-info",
    metavar="CSV_FILE",
    required=True,
    help=(
        "CSV file with NAME, REGION, and SPEED columns to fill missing"
        " information from the jsonl data"
    ),
    type=Path,
)
parser.add_argument(
    "-q",
    "--quiet",
    dest="verbose",
    default=True,
    action="store_false",
    help="show less output",
)
parser.add_argument(
    "-o",
    "--out-dir",
    dest="out_dir",
    default=Path(),
    help="directory to save plots to",
    type=Path,
)
args = parser.parse_args(argv)

for file in chain(args.jsonl_file, (args.node_info,)):
    if not file.is_file():
        parser.error(f"Not a valid file: {file}")

if args.out_dir.exists():
    if not args.out_dir.is_dir():
        parser.error("Output directory already exists but isn't a directory")
else:
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    except OSError:
        parser.error("Couldn't create output directory")

assert args.out_dir.is_dir()


# %% Define script inputs


jsonl_files: Iterable[Path] = args.jsonl_file
out_dir: Path = args.out_dir
node_info_file = args.node_info
verbose: bool = args.verbose


# %% Data loading routines


# Filter to these columns and coerce to these dtypes
columns = {
    "nodename": str,
    "region": str,
    "link_speed": str,
    "cpu_name": str,
    "gpu_name": str,
    "gpu_gb": int,
    "scheme": str,
    "duration": float,
    "total_bytes_read": int,
    "rate": float,
    "source": str,
    "raw_read": bool,
    "force_http": bool,
    "lazy_load": bool,
    "plaid_mode": bool,
    "plaid_buffers": int,
    "verify_hash": bool,
    "cached": bool,
    "read_size": int,
    "buffer_size": int,
}

bool_columns = {k: v for k, v in columns.items() if v is bool}


def read_csv(filename: str) -> pd.DataFrame:
    # Use read_jsonl instead of this function if possible;
    # This one has not been tested as thoroughly.
    return pd.read_csv(
        filename,
        low_memory=False,
        usecols=list(columns.keys()),
        dtype=bool_columns,
    )


def filter_jsonl(path: Union[str, os.PathLike]) -> StringIO:
    # Filter any extraneous logs from the output file
    filtered = StringIO()
    with open(path, "r") as file:
        filtered.writelines(
            (
                line
                for line in file
                if line.startswith("{") and line.rstrip().endswith("}")
            )
        )
    filtered.seek(0)
    return filtered


def read_jsonl(filename: Union[str, os.PathLike]) -> pd.DataFrame:
    with filter_jsonl(filename) as jsonl_data:
        return pd.read_json(jsonl_data, lines=True, dtype=bool_columns)[
            columns.keys()
        ]


# %% Load files


data = pd.concat(map(read_jsonl, jsonl_files))

# Disable warnings
pd.options.mode.chained_assignment = None

if verbose:
    print(data.head())


# %% Fixing dtypes for fully-null columns


# Pandas assumes columns with only null values are of the float64 type,
# so we cast them to a generic object type instead to be able to store strings.
null_columns = {
    column: "object"
    for column, typ in columns.items()
    if typ is not bool and data[column].isnull().all()
}
data = data.astype(null_columns)

if verbose and null_columns:
    print(
        "Column(s)",
        ", ".join(null_columns),
        "were null; changed to generic object type",
    )
    print(data.dtypes)


# %% Load node information from a CSV


node_info_columns = {
    "name": str,
    "region": "category",
    "speed": "category",
}

# Load and filter to relevant columns
node_info = pd.read_csv(node_info_file)[[c.upper() for c in node_info_columns]]
node_info.columns = node_info.columns.str.lower()

# Filter to relevant nodes
node_info = node_info.loc[node_info["name"].isin(data["nodename"])]
node_info.reset_index(drop=True, inplace=True)

# Switch region and speed to categorical variables
node_info = node_info.astype(node_info_columns)

if verbose:
    print(node_info.head())


# %% Merge node information into the main dataframe


def update_node_info(row):
    mask = data["nodename"] == row["name"]
    if row["region"]:
        data.loc[mask, "region"] = row["region"]
    if row["speed"]:
        data.loc[mask, "link_speed"] = row["speed"]


node_info.apply(update_node_info, axis=1)

# Switch region and speed to categorical variables in the main dataframe as well
data = data.astype({"region": "category", "link_speed": "category"})

if verbose:
    print(data.head())


# %% Show counts for some collected data


def format_counts(series) -> str:
    return ", ".join(f"{k}: {v}" for k, v in Counter(series).items())


if verbose:
    for column in "verify_hash", "force_http", "scheme":
        print(column.rjust(14), "-", format_counts(data[column]))


# %% Display information about the GPUs and CPUs present in the data


if verbose:
    for column in "gpu_name", "cpu_name":
        print(column.rjust(14), "-", format_counts(data[column]))


# %% Define some abbreviations for GPU and CPU model names


gpu_nicknames = {
    "NVIDIA RTX A6000": "A6000",
    "NVIDIA RTX A5000": "A5000",
    "NVIDIA RTX A4000": "A4000",
    "NVIDIA A100-SXM4-80GB": "A100(80)",
    "NVIDIA A100-SXM4-40GB": "A100(40)",
    "NVIDIA A100 80GB PCIe": "A100(P80)",
    "NVIDIA A100-PCIE-40GB": "A100(P40)",
    "NVIDIA A40": "A40",
    "Quadro RTX 4000": "Quadro RTX 4000",
    "Quadro RTX 5000": "Quadro RTX 5000",
    "Tesla V100-SXM2-16GB": "V100",
}

cpu_nicknames = {
    "AMD EPYC 7532 32-Core Processor": "EPYC-7532",
    "AMD EPYC 7513 32-Core Processor": "EPYC-7513",
    "AMD EPYC 7413 24-Core Processor": "EPYC-7413",
    "Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz": "Intel-8358",
    "Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz": "Intel-6134",
    "Intel(R) Xeon(R) Silver 4208 CPU @ 2.10GHz": "Intel-4208",
    "AMD Ryzen Threadripper 3970X 32-Core Processor": "Ryzen-3970X",
    "AMD EPYC 74F3 24-Core Processor": "EPYC-74F3",
    "AMD EPYC 7443P 24-Core Processor": "EPYC-7443P",
    "AMD EPYC 7402P 24-Core Processor": "EPYC-7402P",
    "AMD EPYC 7352 24-Core Processor": "EPYC-7352",
    "AMD EPYC 7282 16-Core Processor": "EPYC-7282",
}


nickname_warnings = set()


def nickname(long_name: str) -> str:
    for nickname_mapping in gpu_nicknames, cpu_nicknames:
        if long_name in nickname_mapping:
            return nickname_mapping[long_name]
    else:
        if verbose and long_name not in nickname_warnings:
            nickname_warnings.add(long_name)
            print(f"Warning: missing nickname entry for {long_name}")
        return long_name


data["cpu_nickname"] = data["cpu_name"].apply(nickname).astype("category")
data["gpu_nickname"] = data["gpu_name"].apply(nickname).astype("category")


# %% Normalize plaid_buffers to ints with 0 representing "not plaid_mode"


if "plaid_buffers" in data.columns:
    data["plaid_buffers"] = data["plaid_buffers"].fillna(0).astype(int)


# %% Define a generic labelling system


def active_labels(specifiers: Iterable[Union[Callable, str]], row):
    labels = []

    for specifier in specifiers:
        if isinstance(specifier, str):
            labels.append(str(row[specifier]))
        else:
            result = specifier(row)
            if result:
                labels.append(str(result))

    return "\n".join(labels) or "default"


def curry(f):
    return partial(partial, f)


@curry
def get_param(param_name: str, row) -> str:
    return f"{param_name}={row[param_name]}"


@curry
def when_true(column_name: str, row) -> str:
    return column_name if row[column_name] else None


@curry
def if_present(column_name: str, row) -> str:
    return row[column_name] if column_name in row and row[column_name] else None


label_callback = curry(active_labels)


def apply_labels(specifiers, dataset=data):
    dataset.loc[:, "mode"] = dataset.apply(
        label_callback(specifiers),
        axis=1,
    ).astype("category")


# %% Utility functions for plotting


mebibyte, gibibyte = 1 << 20, 1 << 30


def minmax(series: Iterable) -> tuple:
    return min(series), max(series)


@lru_cache(maxsize=4)
def get_tick_marks(
    limits: tuple,
    step: int = 128 * mebibyte,
    scale: int = gibibyte,
) -> Tuple[range, Iterable[str]]:
    min_t, max_t = limits
    lower_bound = int(min_t - (min_t % step))
    upper_bound = int(max_t + (-max_t % step))
    tick_marks = range(lower_bound, upper_bound + 1, step)
    return tick_marks, [f"{v / scale:.3f}" for v in tick_marks]


def set_sns_style(font_size=0.68, **kwargs) -> None:
    sns.set_style("darkgrid")
    sns.set_context("paper")
    sns.set(font_scale=font_size, **kwargs)


def remove_legend(ax: plt.Axes) -> None:
    if hasattr(ax, "legend_") and hasattr(ax.legend_, "remove"):
        ax.legend_.remove()


def save_figure(
    figure: plt.Figure,
    filename: Union[str, os.PathLike],
    margins: Optional[dict] = None,
    fallback_margins: Optional[dict] = None,
) -> None:
    if not fallback_margins:
        fallback_margins = {}
    if not margins:
        margins = fallback_margins
    else:
        margins = {**fallback_margins, **margins}
    figure.subplots_adjust(**margins)
    figure.savefig(filename, dpi=300)
    figure.clf()
    plt.clf()
    plt.close(figure)


# %% Define a letter-value plot convenience function with nice defaults


def plot(
    sample: pd.DataFrame,
    title: str,
    filename: Optional[Union[str, os.PathLike]] = None,
    margins: Optional[dict] = None,
    extra_text: Optional[str] = None,
    hue=None,
    *,
    order: Union[Iterable[str], None, Literal["reverse"]] = "reverse",
    suppress_legend: bool = False,
    x_column: str = "rate",
    x_label: str = "deserialization speed (GiB / sec)",
    y_column: str = "mode",
    y_label: str = "mode",
    x_tick_rotation: float = -75,
    save: bool = True,
):
    set_sns_style()

    bonus = {"hue": hue} if hue is not None else {}
    if order == "reverse":
        order = list(reversed(sorted(sample[y_column].dtype.categories)))

    ax = sns.boxenplot(
        sample, x=x_column, y=sample[y_column], order=order, **bonus
    )

    # Side effect of hue
    if suppress_legend:
        remove_legend(ax)

    ax.set(xlabel=x_label, ylabel=y_label)
    fig: plt.Figure = ax.get_figure()
    fig.suptitle(title)

    # Set tick marks
    positions, labels = get_tick_marks(limits=minmax(sample[x_column]))
    ax.set_xticks(positions, labels=labels, rotation=x_tick_rotation)

    if extra_text:
        fig.text(0, 0, extra_text)

    if save:
        if not filename:
            raise ValueError("Must specify filename when saving")
        fallback_margins = dict(bottom=0.15, left=0.32, top=0.95, right=0.95)
        save_figure(fig, filename, margins, fallback_margins)
    else:
        return ax


# %% Heavily filtered, simple plot


def simple_plot(
    filename: Optional[Union[str, os.PathLike]] = (
        out_dir / "tensorizer-deserialization"
    ),
):
    # Filter to the fewest variables possible
    filtered_schemes = ("s3", "redis", "s3s", "https")
    # 16 MiB buffer size, no s3/redis/https, no hash verification;
    # only local files,
    # the HTTP responses that successfully hit a cache,
    # and raw network reads
    mask = (
        (data["buffer_size"] == 16 * mebibyte)
        & np.in1d(data["scheme"], filtered_schemes, invert=True)
        & ~(data["verify_hash"])
        & (
            (data["scheme"] != "http")
            | data["cached"].astype(bool)
            | data["raw_read"]
        )
    )

    if "plaid_buffers" in data.columns:
        # Filter to non-plaid mode or plaid mode with two buffers
        mask &= np.in1d(data["plaid_buffers"], [0, 2])

    # All torch results are merged in at this point because the previous filters
    # on tensorizer-specific parameters would otherwise remove them all
    mask |= data["scheme"] == "torch"

    # Then, filter to decently fast machines
    mask &= data["link_speed"] == "40G"
    source = data.loc[mask]

    def deserialization_type(row) -> str:
        scheme = row["scheme"]
        if scheme == "torch":
            return "torch.load()"
        elif row["raw_read"]:
            return "raw network read"
        else:
            return f"tensorizer {scheme}"

    @curry
    def when_not_torch(callback, row) -> Optional[str]:
        return None if row["scheme"] == "torch" else callback(row)

    plt.clf()
    apply_labels(
        (
            deserialization_type,
            when_not_torch(when_true("plaid_mode")),
            when_not_torch(when_true("lazy_load")),
        ),
        source,
    )

    order = [
        "torch.load()",
        "tensorizer file",
        "tensorizer file\nplaid_mode",
        "tensorizer file\nlazy_load",
        "raw network read",
        "tensorizer http",
        "tensorizer http\nplaid_mode",
        "tensorizer http\nlazy_load",
    ]

    if verbose:
        # Print some statistics on the number of samples in each category
        print(source.groupby("mode").count())

    plot(
        source,
        "Deserialization Speeds",
        filename,
        margins=dict(left=0.20, right=0.98),
        order=order,
        save=True,
    )


simple_plot()


# %% Filter to tensorizer data


tensorizer_data = data.dropna(subset="buffer_size").astype({"buffer_size": int})


# %% Define a multi-line plot convenience function with nice defaults


def plot_line_buffer_sizes(
    sample: pd.DataFrame,
    title: Optional[str] = None,
    filename: Optional[Union[str, os.PathLike]] = None,
    margins: Optional[dict] = None,
    hue: Optional[str] = "mode",
    *,
    order: Optional[Iterable[str]] = None,
    x_column: str = "buffer_size",
    x_label: str = "buffer size (MiB)",
    y_column: str = "rate",
    y_label: str = "download speed (GiB / sec)",
    save: bool = True,
):
    set_sns_style()
    grid = sns.catplot(
        sample,
        x=x_column,
        y=y_column,
        kind="point",
        errorbar="sd",
        err_kws={"alpha": 0.6, "linewidth": 1},
        capsize=0.4,
        hue=hue,
        linestyles=["-", "--", ":", "-."],
        order=order,
    )

    x_tick_positions = sample[x_column].unique()
    x_tick_positions.sort()
    x_tick_labels = [
        str(Decimal(int(val)) / mebibyte) for val in x_tick_positions
    ]

    limits = tuple(np.percentile(sample[y_column], (2, 98)))
    positions, labels = get_tick_marks(limits=limits)
    for ax in grid.axes.flat:
        ax.set_yticks(positions, labels=labels)
        ax.set_xticks(ax.get_xticks(), labels=x_tick_labels)

    grid.set(xlabel=x_label, ylabel=y_label)
    fig: plt.Figure = grid.figure

    if title:
        fig.suptitle(title)

    if save:
        if not filename:
            raise ValueError("Must specify filename when saving")
        fallback_margins = dict(left=0.125)
        save_figure(fig, filename, margins, fallback_margins)
    else:
        return grid


# %% Generate a multi-line plot over various subsets of the data


default_schemes = (("http", "https"), ("redis",), ("s3", "s3s"), ("file",))


def multi_buffer_size_plot(
    base_filename: Optional[Union[str, os.PathLike]] = (
        out_dir / "tensorizer-buffer-sizes"
    ),
    schemes: Iterable[Iterable[str]] = default_schemes,
):
    source = tensorizer_data
    # no hash verification;
    # only local files or the HTTP responses that successfully hit a cache
    mask = ~(source["verify_hash"]) & (
        (source["scheme"] != "http") | source["cached"].astype(bool)
    )

    if "plaid_buffers" in source.columns:
        # Filter to non-plaid mode or plaid mode with two buffers
        mask &= np.in1d(source["plaid_buffers"], [0, 2])

    # Filter to decently fast machines
    mask &= source["link_speed"] == "40G"

    for scheme in (None, *schemes):
        plt.clf()
        sample = source.loc[mask]
        filename = Path(base_filename)

        if scheme is None:
            title = "Overall speeds by read buffer size"
        else:
            sample = sample.loc[np.in1d(sample["scheme"], scheme)]
            title = f"Speeds ({scheme[0]}) by read buffer size"
            filename = filename.with_stem("-".join((filename.stem, scheme[0])))

        if sample.empty:
            continue

        apply_labels(
            (
                when_true("plaid_mode"),
                when_true("lazy_load"),
            ),
            sample,
        )
        plot_line_buffer_sizes(
            sample, filename=filename, title=title, save=True
        )


multi_buffer_size_plot()


# %% Define a line plot convenience function with nice defaults


def plot_line_plaid_buffer_count(
    sample: pd.DataFrame,
    title: Optional[str] = None,
    filename: Optional[Union[str, os.PathLike]] = None,
    margins: Optional[dict] = None,
    hue: Optional[str] = None,
    *,
    x_column: str = "plaid_buffers",
    x_label: str = "plaid buffer count",
    y_column: str = "rate",
    y_label: str = "download speed (GiB / sec)",
    save: bool = True,
):
    set_sns_style()
    sample = sample[sample[x_column] >= 1]
    ax = sns.lineplot(
        sample,
        x=x_column,
        y=y_column,
        hue=hue,
    )

    max_x_tick_position = sample[x_column].max()
    x_tick_positions = []
    i = 1
    while i <= max_x_tick_position:
        x_tick_positions.append(i)
        i <<= 1
    ax.set_xticks(x_tick_positions)

    ax.set(xlabel=x_label, ylabel=y_label)
    fig = ax.get_figure()
    if title:
        fig.suptitle(title)

    if save:
        if not filename:
            raise ValueError()
        save_figure(fig, filename, margins)
    else:
        return ax


# %% Generate a line plot over various subsets of the data


def multi_plaid_buffer_count_plot(
    base_filename: Optional[Union[str, os.PathLike]] = (
        out_dir / "tensorizer-plaid-buffer-count"
    ),
    schemes: Iterable[Iterable[str]] = default_schemes,
):
    source = tensorizer_data
    # no hash verification;
    # only local files or the HTTP responses that successfully hit a cache
    mask = ~(source["verify_hash"]) & (
        (source["scheme"] != "http") | source["cached"].astype(bool)
    )

    # Filter to samples that use plaid buffers
    mask &= source["plaid_buffers"] >= 1

    # Filter to decently fast machines
    mask &= source["link_speed"] == "40G"

    for scheme in (None, *schemes):
        plt.clf()
        sample = source.loc[mask]
        filename = Path(base_filename)

        if scheme is None:
            title = "Overall speeds by plaid buffer count"
        else:
            sample = sample.loc[np.in1d(sample["scheme"], scheme)]
            title = f"Speeds ({scheme[0]}) by plaid buffer count"
            filename = filename.with_stem("-".join((filename.stem, scheme[0])))

        if sample.empty:
            continue

        plot_line_plaid_buffer_count(
            sample, filename=filename, title=title, save=True
        )


multi_plaid_buffer_count_plot()


# %% Plot GPU and CPU distributions used in the tests


def plot_dist(
    sample: pd.DataFrame,
    column: str,
    order: Optional[Iterable[str]] = None,
    filename: Union[str, os.PathLike] = out_dir / "distribution",
    title: Optional[str] = None,
    margins: Optional[dict] = None,
):
    counts = Counter(sample[column])
    ax = sns.barplot(counts, order=order, orient="h")
    fig: plt.Figure = ax.get_figure()
    if title:
        fig.suptitle(title)
    plt.subplots_adjust()
    fallback_margins = dict(bottom=0.08, left=0.28, top=0.95, right=0.95)
    save_figure(fig, filename, margins, fallback_margins)


def dist_order(order: Iterable[str], rest: pd.Series):
    # Give priority to the ordering from the first argument,
    # but filter down to those present in the second argument,
    # and add any that are missing
    order = dict.fromkeys(order)
    present = set(rest.unique())
    for not_present in order.keys() - present:
        del order[not_present]
    not_represented = dict.fromkeys(sorted(present - order.keys()))
    order.update(not_represented)
    return order.keys()


def plot_gpu_cpu_dists():
    plot_dist(
        data,
        column="gpu_name",
        order=dist_order(gpu_nicknames.keys(), data["gpu_name"]),
        filename=out_dir / "tensorizer-benchmark-gpu-distribution",
        title="GPU Distribution",
    )

    plot_dist(
        data,
        column="cpu_name",
        order=dist_order(cpu_nicknames.keys(), data["cpu_name"]),
        filename=out_dir / "tensorizer-benchmark-cpu-distribution",
        title="CPU Distribution",
        margins=dict(left=0.485),
    )


plot_gpu_cpu_dists()

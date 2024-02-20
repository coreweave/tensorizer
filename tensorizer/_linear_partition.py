from typing import Iterable, List, Sequence

import numpy as np


__all__ = ("partition",)


def partition(
    weights: Sequence[int],
    partitions: int,
    prefer_fewer: bool = True,
    performance_threshold: int = 250000,
) -> Iterable[slice]:
    """
    Partitions a sequence of weights into slices with balanced sums,
    without changing the ordering of elements.
    Balancing minimizes the largest sum of any resulting slice.
    Args:
        weights: Element weights to balance.
        partitions: The maximum number of slices to return.
            May return fewer if there are too few weights.
        prefer_fewer: Enables returning fewer slices than requested
            if the optimal partitioning scheme with at most `partitions`
            partitions does not need exactly `partitions` slices.
            For example, asking for 3 slices of ``[5, 2, 2]``
            could return ``[[5], [2, 2]]``, since splitting again
            to ``[[5], [2], [2]]`` doesn't improve the largest slice's sum.
        performance_threshold: Limit for the number of comparisons that would
            be used to calculate an optimal partitioning scheme. If more than
            this many comparisons would be needed, an asymptotically faster
            greedy approximation is used instead.

    Returns:
        An iterable of ``slice`` objects denoting ranges of the original
        sequence that belong to each partition.

    Examples:
        Balancing a list of integers::

            ints = [2, 8, 10, 5, 5]
            split = [ints[s] for s in partition(ints, 3)]
            assert split == [[2, 8], [10], [5, 5]]

        Balancing a list of strings by length::

            strings = ["abc", "ABC", "12345", "XYZ", "654321"]
            weights = [len(s) for s in strings]
            split = [strings[s] for s in partition(weights, 4)]
            assert split == [["abc", "ABC"], ["12345"], ["XYZ"], ["654321"]]
    """
    n: int = len(weights)
    partitions = min(n, partitions)
    if (
        partitions <= 2
        or partitions * (n * (n + 1) // 2) > performance_threshold
    ):
        return greedy_linear_partition(weights, partitions)
    else:
        return linear_partition(weights, partitions, prefer_fewer)


def linear_partition(
    weights: Sequence[int], partitions: int, prefer_fewer: bool = True
) -> Iterable[slice]:
    # Dynamic programming solution to the linear partitioning problem
    # based on Dr. Steven Skiena's algorithm & lecture notes.
    # This finds an exact, optimal solution that has the smallest maximum sum
    # in any partition. It provides no guarantees that the other partitions
    # are perfect balanced, as long as they each have smaller sums than
    # the largest partition.
    # Time complexity: O(partitions * (len(weights) ** 2))
    # Space complexity: O(partitions * len(weights))
    n: int = len(weights)
    partitions = min(partitions, n)
    if partitions <= 1:
        return (slice(0, n),)
    prefix_sums: np.ndarray = np.cumsum(weights, dtype=np.uint64)
    inf: np.ndarray = np.array(-1).astype(np.uint64)
    # Entries in best_prefix_partition represent the lowest cost for a prefix,
    # given two parameters:
    # - Row: end position of the prefix
    # - Column: number of allowed new partitions
    # The lowest cost is the size of the largest partition in that prefix
    # assuming the partitions are balanced optimally
    best_prefix_partition: np.ndarray = np.full(
        (n, partitions), inf, dtype=np.uint64
    )

    # Each row represents a different prefix
    # No new partitions means no splitting; the total weight is the whole prefix
    best_prefix_partition[:, 0] = prefix_sums

    # Each column represents splitting into a different number of partitions
    # Since the first row is a prefix with only one item,
    # regardless of the number of partitions, it will have the same maximum
    # weight (that single weight)
    best_prefix_partition[0, :] = weights[0]

    partition_starts: np.ndarray = np.empty(
        (n - 1, partitions - 1), dtype=np.uint64
    )
    # The columns in partition_starts mean the same as the columns in
    # best_prefix_partition, except row & column 0 would be meaningless,
    # so we remove them, since there's nothing to end if there are no new
    # partitions allowed, and nothing precedes position 0.
    # The rows in partition_starts are a lookup table
    # An entry at partition_starts[end, k] being `start` means the optimal
    # partition ending immediately before `end` starts at `start`,
    # with k other partitions behind it.
    # The final partition indices come from iterated application of
    # looking up `end` in partition_starts, and replacing it with the result
    for end in range(1, n):
        for partitions_remaining in range(1, partitions):
            for start in range(end):
                # If you form a new partition between `start` and `end`,
                # then the largest sum overall is either the new one,
                # or the previous largest sum in everything preceding `start`
                # (with fewer allowed partitions)
                largest_partition = max(
                    best_prefix_partition[start, partitions_remaining - 1],
                    prefix_sums[end] - prefix_sums[start],
                )
                best = best_prefix_partition[end, partitions_remaining]
                if (
                    prefer_fewer
                    and largest_partition < best
                    or not prefer_fewer
                    and largest_partition <= best
                ):
                    # If forming a partition here gave
                    # a record low outcome, record it.
                    # If prefer_fewer is true, this may result in fewer
                    # partitions than originally requested, if adding more
                    # wouldn't reduce the maximum partition size either way
                    best_prefix_partition[end, partitions_remaining] = (
                        largest_partition
                    )
                    partition_starts[end - 1, partitions_remaining - 1] = start
    del best_prefix_partition
    slices = []
    end = n - 1
    for partitions_remaining in range(partitions - 1, 0, -1):
        start = int(partition_starts[end - 1, partitions_remaining - 1])
        # The algorithm calculates an inclusive endpoint,
        # so end at end + 1 to include it, and start at start + 1
        # to not overlap with the next slice
        slices.append(slice(start + 1, end + 1))
        end = start
        if end == 0:
            break
    slices.append(slice(0, end + 1))
    slices.reverse()

    return slices


def greedy_linear_partition(
    weights: Sequence[int], partitions: int
) -> Iterable[slice]:
    # Greedy approximation for the linear partitioning problem, adapted from:
    # https://www.werkema.com/2021/11/01/an-efficient-solution-to-linear-partitioning/
    # Time complexity: O(len(weights))
    # Space complexity: O(partitions)
    # Could have O(1) space if changed to be a generator
    partitions = min(len(weights), partitions)
    if partitions <= 1:
        return (slice(0, len(weights)),)

    # This implementation scales weights by 2 * partitions to avoid fractions
    target_size: int = sum(weights) * 2
    current_size: int = 0

    groups: List[slice] = []
    start: int = 0

    for end, weight in enumerate(weights):
        scaled_weight: int = weight * partitions
        current_size += scaled_weight * 2
        if current_size > target_size + scaled_weight:
            groups.append(slice(start, end))
            start = end
            if len(groups) == partitions - 1:
                break
            current_size -= target_size

    groups.append(slice(start, len(weights)))
    return groups

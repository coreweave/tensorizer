import itertools
import sys
from typing import Iterable, List, Sequence, Tuple

__all__ = ("partition",)


def partition(
    weights: Sequence[int],
    partitions: int,
    performance_threshold: int = 100,
) -> Iterable[slice]:
    """
    Partitions a sequence of weights into slices with balanced sums,
    without changing the ordering of elements.
    Balancing minimizes the largest sum of any resulting slice.
    Args:
        weights: Element weights to balance.
        partitions: The maximum number of slices to return.
            May return fewer if there are too few weights.
        performance_threshold: Limit on the estimated time that would
            be required to calculate an optimal partitioning scheme.
            Not an exact measurement, but similar to milliseconds,
            with an additional fuzzy bound on memory usage thrown in.
            If this threshold is passed, an asymptotically faster,
            low-memory greedy approximation is used instead.

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
    # The strange formula came from a least-squares fit over testing data;
    # it is likely an overestimate, but approximates the time in milliseconds
    # to run linear_partition
    too_intensive: bool = (
        round(
            max(
                (partitions * n) / 5000,
                n**1.6565 * partitions**0.617 * 1.75e-4,
            )
        )
        > performance_threshold
    )
    if partitions <= 2 or too_intensive:
        return greedy_linear_partition(weights, partitions)
    else:
        return linear_partition(weights, partitions)


def linear_partition(
    weights: Sequence[int], partitions: int
) -> Iterable[slice]:
    n: int = len(weights)
    partitions = min(partitions, n)
    if partitions <= 1:
        return (slice(0, n),)
    for w in weights:
        if w < 0:
            raise ValueError("All weights must be non-negative")
    prefix_sums: Tuple[int, ...] = tuple(
        itertools.accumulate(weights, initial=0)
    )

    inf: int = prefix_sums[-1] + 1
    sentinel = (inf, inf)
    memo = [sentinel] * ((n + 1) * partitions)
    # Key function: end * partitions + preceding_parts
    null = (0, 0)
    for i in range(partitions):
        # When end = 0
        memo[i] = null
    del null
    for i in range(n + 1):
        # When preceding_parts = 0
        memo[i * partitions] = (0, prefix_sums[i])

    def find_start(end: int, preceding_parts: int) -> Tuple[int, int]:
        key = end * partitions + preceding_parts
        cache_hit = memo[key]
        if cache_hit is not sentinel:
            return cache_hit

        best_weight = inf
        best_start = -1
        end_sum = prefix_sums[end]

        # Optimization: In general, iterating in reverse will find the best one
        # faster for "uniformly shuffled" datasets, since the best segment split
        # will likely be closer to (end / preceding_parts) in length.
        # There are two more important observations:
        # 1. current_weight is monotonically increasing while moving
        #    right-to-left, because the current segment is expanding
        # 2. earlier_weight is monotonically decreasing while moving
        #    right-to-left, because the preceding segment is shrinking
        #
        # This means there are two regions that can be completely skipped:
        # 1. current_weight is too big (> best_weight)
        #   - This means the start is too far left
        # 2. earlier_weight is too big (> best_weight)
        #   - This means the start is too far right
        # Since best_weight is monotonically decreasing, regions skipped like
        # this never need to be revisited. On the other hand, each time
        # best_weight updates, more may be eligible to be skipped.
        # The sooner good candidates are found for best_weight,
        # the sooner the search space will be narrowed.
        # This leads to the following strategy:
        # 1. Keep track of the left and right boundaries of the search space
        # 2. While the rightmost element is a new best, shrink the right by 1
        # 3. When the rightmost element is not a new best:
        #   a) Shrink the right by 1, then
        #   b) Jump half of the remaining search space towards the left
        #   c) If current_weight is now too big, update the left,
        #      then jump half of the new search space back towards the right
        #   d) If earlier_weight is still too big, update the right to the
        #      jumped-to position, and then go back to step 3
        #   e) If earlier_weight is not too big, update best_weight,
        #      jump back to the right, and go back to step 2
        # This attempts to find the valid region for current_weight
        # and earlier_weight as quickly as possible via a dynamic variant of
        # binary search, and then linearly scans through the possibilities.

        left = 0  # first impossible element due to current_weight
        # Note: start = 0 is reserved for when preceding_parts = 0 anyway,
        # which is always handled by the cache, so setting left = 0 is safe.
        right = end - 1  # last possible element due to earlier_weight
        start = right
        while True:
            current_weight = end_sum - prefix_sums[start]
            if current_weight < best_weight:
                earlier_weight = find_start(start, preceding_parts - 1)[1]
                if earlier_weight < best_weight:
                    best_weight = (
                        current_weight
                        if current_weight > earlier_weight
                        else earlier_weight
                    )
                    best_start = start
                    # If this was already the rightmost one, narrow the search
                    right -= start == right
                    if right <= left:
                        break
                    # Reset to the right end, in case the best was skipped
                    start = right
                else:
                    # Nothing right of this matters
                    # Try skipping forward a bit
                    right = start - 1
                    dist = right - left
                    if dist <= 0:
                        break
                    else:
                        start = right - (dist >> 1)
            else:
                # Overshot, nothing left of this matters
                left = start
                dist = right - left
                if dist <= 0:
                    break
                elif dist == 1:
                    start = right
                else:
                    start = left + (dist >> 1)

        result = (best_start, best_weight)
        memo[key] = result
        return result

    if partitions > 900:
        old_recursion_limit = sys.getrecursionlimit()
        # Loosen the recursion limit by up to about 6500 if needed.
        # Since too-high limits can cause the interpreter to crash,
        # anything beyond this point is handled on a purely algorithmic level.
        sys.setrecursionlimit(old_recursion_limit + min(partitions, 6500) + 10)
    else:
        old_recursion_limit = None
    try:
        for parts_before in range(6500, partitions - 1, 6500):
            # Limit the stack depth by pre-populating the cache
            # for extremely high numbers of partitions (> 6500).
            # Despite caching, this can easily take more time than
            # the main call, because it computes several extra values
            # that would have normally been skipped.
            parts_after: int = partitions - parts_before - 1
            # n - parts_after is the closest point to the end that could
            # feasibly have parts_after parts after it, skipping impossible
            # scenarios like end=n, parts_before=0
            for i in range(0, n - parts_after + 1):
                find_start(i, parts_before)

        i = n
        seq = [n]
        for k in range(1, partitions):
            i = find_start(i, partitions - k)[0]
            if i == 0:
                break
            seq.append(i)
        seq.append(0)
        seq.reverse()
    finally:
        if old_recursion_limit is not None:
            sys.setrecursionlimit(old_recursion_limit)
    memo.clear()
    return tuple(slice(a, b) for a, b in zip(seq, seq[1:]))


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

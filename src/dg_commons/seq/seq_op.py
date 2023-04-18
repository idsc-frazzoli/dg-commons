from operator import add

import numpy as np
from toolz import accumulate

from dg_commons.seq.sequence import DgSampledSequence, iterate_with_dt, Timestamp

__all__ = ["seq_accumulate", "seq_integrate", "seq_differentiate"]


# todo check that this operations cannot be done on any values


def seq_integrate(sequence: DgSampledSequence[float]) -> DgSampledSequence[float]:
    """Integrates with respect to time - multiplies the value with delta T."""
    if not sequence:
        msg = "Cannot integrate empty sequence."
        raise ValueError(msg)
    # if sequence is made up by a single value and timestamp
    if len(sequence.values) == 1:
        return sequence
    total = 0.0
    timestamps = []
    values = []
    for it in iterate_with_dt(sequence):
        v_avg = (it.v0 + it.v1) / 2.0
        total += v_avg * it.dt
        timestamps.append(it.t1)
        values.append(total)

    return DgSampledSequence[float](timestamps, values)


def seq_differentiate(sequence: DgSampledSequence[float]) -> DgSampledSequence[float]:
    """Differentiate with respect to time."""
    if not sequence:
        msg = "Cannot differentiate empty sequence."
        raise ValueError(msg)
    timestamps = []
    values = []
    for it in iterate_with_dt(sequence):
        derivative = (it.v1 - it.v0) / it.dt
        timestamps.append(it.t0 + it.dt / 2.0)
        values.append(derivative)
    return DgSampledSequence[float](timestamps, values)


def seq_accumulate(sequence: DgSampledSequence[float]) -> DgSampledSequence[float]:
    cumsum = list(accumulate(add, sequence.values))
    return DgSampledSequence[sequence.XT](sequence.timestamps, cumsum)


def find_crossings(sequence: DgSampledSequence[float], threshold: float) -> list[Timestamp]:
    """Finds the timestamps where the sequence crosses the threshold."""
    values = np.array(sequence.values) - threshold
    zero_crossings = np.where(np.diff(np.signbit(values)))[0]
    crossings: list[Timestamp] = []
    for i in zero_crossings:
        i_plus_1 = i + 1
        t0, t1 = sequence.timestamps[i], sequence.timestamps[i_plus_1]
        v0, v1 = values[i], values[i_plus_1]
        dt = t1 - t0
        # linear interpolation
        t = t0 + (v0 / (v0 - v1)) * dt
        crossings.append(t)
    return crossings

from operator import add

from toolz import accumulate

from dg_commons.seq.sequence import DgSampledSequence, iterate_with_dt

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

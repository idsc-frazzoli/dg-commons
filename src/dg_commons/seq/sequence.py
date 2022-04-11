from bisect import bisect_right, bisect_left
from dataclasses import dataclass, InitVar, field
from decimal import Decimal as D
from typing import Generic, TypeVar, List, Callable, Type, Iterator, Union, get_args, Any, Sequence, Tuple

from zuper_commons.types import ZException, ZValueError

__all__ = [
    "Timestamp",
    "DgSampledSequence",
    "DgSampledSequenceBuilder",
    "IterateDT",
    "iterate_with_dt",
    "UndefinedAtTime",
]

from dg_commons import DgCommonsConstants

X = TypeVar("X")
Y = TypeVar("Y")
Timestamp = Union[D, float, int]


class UndefinedAtTime(ZException):
    pass


@dataclass(unsafe_hash=True)
class DgSampledSequence(Generic[X]):
    """A sampled time sequence. Only defined at certain points.
    Modernized version of the original SampledSequence from Duckietown:
    https://github.com/duckietown/duckietown-world/blob/daffy/src/duckietown_world/seqs/tsequence.py
    Modification:
        - It is hashable
        - Adds the possibility of interpolating
        - removing possibility of assigning post-init timestamps and values fields
        - Seamless support for different timestamps types (e.g. float or Decimal)
    """

    timestamps: InitVar[Sequence[Timestamp]]
    values: InitVar[Sequence[X]]

    _timestamps: Tuple[Timestamp] = field(default_factory=tuple)
    _values: Tuple[X] = field(default_factory=tuple)

    def __post_init__(self, timestamps, values):
        if DgCommonsConstants.checks:
            if len(timestamps) != len(values):
                raise ZValueError("Length mismatch of Timestamps and values")

            for t in timestamps:
                if not isinstance(t, get_args(Timestamp)):
                    raise ZValueError(f'I expected a real number as "Timestamp", got {type(t)}')
            for i in range(len(timestamps) - 1):
                dt = timestamps[i + 1] - timestamps[i]
                if dt <= 0:
                    raise ZValueError(f"Invalid dt = {dt} at i = {i}; ts= {timestamps}")
            ts_types = set([type(ts) for ts in timestamps])
            if D in ts_types and any([int in ts_types, float in ts_types]):
                raise ZValueError(
                    "Attempting to create SampledSequence with mixed Decimal and floats", timestamps=timestamps
                )
        self._timestamps = tuple(timestamps)
        self._values = tuple(values)

    @property
    def XT(self) -> Type[X]:
        return get_args(self.__orig_class__)[0]

    @property
    def timestamps(self) -> Tuple[Timestamp]:
        return self._timestamps

    @timestamps.setter
    def timestamps(self, v: Any) -> None:
        raise RuntimeError("Cannot set timestamps of SampledSequence directly")

    @property
    def values(self) -> Tuple[X]:
        return self._values

    @values.setter
    def values(self, v: Any) -> None:
        raise RuntimeError("Cannot set values of SampledSequence directly")

    def at(self, t: Timestamp) -> X:
        """Returns value at requested timestamp, raises UndefinedAtTime if not defined at t"""
        try:
            i = self._timestamps.index(t)
            return self._values[i]
        except ValueError:
            msg = f"Could not find Timestamp: {t} in: {self._timestamps}"
            raise UndefinedAtTime(msg)

    def at_or_previous(self, t: Timestamp) -> X:
        """
        @:return: Value at requested timestamp or at previous.
        @:raise UndefinedAtTime if there is no previous
        """
        if t < self.get_start():
            msg = f"Could not find at_or_previous with Timestamp: {t} in: {self._timestamps}"
            raise UndefinedAtTime(msg)
        try:
            return self.at(t)
        except UndefinedAtTime:
            i = bisect_right(self._timestamps, t)
            return self._values[i - 1]

    def at_interp(self, t: Timestamp) -> X:
        """Interpolates between timestamps, holds at the extremes
        @:return: Value at requested timestamp.
        """
        if t <= self.get_start():
            return self._values[0]
        elif t >= self.get_end():
            return self._values[-1]
        else:
            i = bisect_right(self._timestamps, t)
            scale = (float(t) - float(self._timestamps[i - 1])) / (float(self._timestamps[i] - self._timestamps[i - 1]))
            return self._values[i - 1] * (1 - scale) + self._values[i] * scale

    def get_start(self) -> Timestamp:
        """
        @:return: The timestamp for start
        """
        if not self._timestamps:
            raise ZValueError("Empty sequence")
        return self._timestamps[0]

    def get_end(self) -> Timestamp:
        """
        @:return: The timestamp for start
        """
        if not self._timestamps:
            raise ZValueError("Empty sequence")
        return self._timestamps[-1]

    def get_sampling_points(self) -> Tuple[Timestamp]:
        """
        Redundant with .timestamps
        @:return: The lists of sampled timestamps
        """
        return self._timestamps

    def transform_values(self, f: Callable[[X], Y], YT: Type[Y] = object) -> "DgSampledSequence[Y]":
        values = []
        timestamps = []
        for t, _ in self:
            res = f(_)
            if res is not None:
                values.append(res)
                timestamps.append(t)
        return DgSampledSequence[YT](timestamps, values)

    def get_subsequence(self, from_ts: Timestamp, to_ts: Timestamp) -> "DgSampledSequence[X]":
        """
        :param from_ts:
        :param to_ts:
        :return: A new sequence with the values between t_start and t_end (extrema included)
        """
        assert from_ts <= to_ts, f"Required t_start <= t_end, got {from_ts},{to_ts}."
        from_idx = bisect_left(self._timestamps, from_ts)
        to_idx = bisect_right(self._timestamps, to_ts)
        return DgSampledSequence[X](timestamps=self._timestamps[from_idx:to_idx], values=self._values[from_idx:to_idx])

    def shift_timestamps(self, dt: Timestamp) -> "DgSampledSequence[X]":
        """
        :param dt:
        :return: A new sequence with the timestamps shifted by dt
        """
        timestamps = [t + dt for t in self.timestamps]
        return DgSampledSequence[X](timestamps=timestamps, values=self.values)

    def __iter__(self):
        return zip(self._timestamps, self._values).__iter__()

    def __len__(self) -> int:
        return len(self._timestamps)


@dataclass
class IterateDT(Generic[X]):
    t0: Timestamp
    t1: Timestamp
    dt: Timestamp
    v0: X
    v1: X


def iterate_with_dt(sequence: DgSampledSequence[X]) -> Iterator[IterateDT[X]]:
    """Yields t0, t1, dt, v0, v1
    Note that timestamps and time deltas are converted to floats for ease of operations
    """
    timestamps = sequence.timestamps
    values = sequence.values
    for i in range(len(timestamps) - 1):
        t0 = float(timestamps[i])
        t1 = float(timestamps[i + 1])
        v0 = values[i]
        v1 = values[i + 1]
        dt = t1 - t0
        yield IterateDT[sequence.XT](t0, t1, dt, v0, v1)


DgSampledSequenceType = TypeVar("DgSampledSequenceType", bound="DgSampledSequence")


@dataclass
class DgSampledSequenceBuilder(Generic[X]):
    timestamps: List[Timestamp] = field(default_factory=list)
    values: List[X] = field(default_factory=list)
    sampled_sequence_type: DgSampledSequenceType = DgSampledSequence

    def add(self, t: Timestamp, v: X):
        if self.timestamps:
            if t <= self.timestamps[-1]:
                msg = "Repeated time stamp"
                raise ZValueError(msg, t=t, timestamps=self.timestamps)
        self.timestamps.append(t)
        self.values.append(v)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def XT(self) -> Type[X]:
        return get_args(self.__orig_class__)[0]

    def as_sequence(self) -> DgSampledSequence:
        return self.sampled_sequence_type[self.XT](timestamps=self.timestamps, values=self.values)

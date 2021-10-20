import itertools
from typing import Callable, Collection, Dict, FrozenSet, Iterator, Mapping, TypeVar

from cytoolz import keyfilter, valmap as valmap_
from frozendict import frozendict

K = TypeVar("K")
V = TypeVar("V")
W = TypeVar("W")

__all__ = ["iterate_dict_combinations", "fkeyfilter", "valmap", "fvalmap", "fd", "fs"]


def iterate_dict_combinations(a: Mapping[K, Collection[V]]) -> Iterator[Mapping[K, V]]:
    """
    Iterates all possible combination. The input is a dictionary where each "player"
    has a collection of "options".
    """
    ks = list(a)
    vs = [a[_] for _ in ks]
    alls = list(itertools.product(*tuple(vs)))
    for x in alls:
        d = frozendict(zip(ks, x))
        yield d


def fkeyfilter(pred: Callable[[K], bool], a: Mapping[K, V]) -> Mapping[K, V]:
    """Wrapper around `cytoolz.keyfilter`. Adds frozendict, and helps with type inference."""
    return frozendict(keyfilter(pred, a))


# def fvalfilter(pred: Callable[[V], bool], a: Mapping[K, V]) -> Mapping[K, V]:
#     """ Wrapper around `cytoolz.valfilter`. Adds frozendict, and helps with types."""
#     return frozendict(valfilter(pred, a))


def fvalmap(pred: Callable[[V], W], a: Mapping[K, V]) -> Mapping[K, W]:
    """Wrapper around `cytoolz.valmap`. Adds frozendict, and helps with type inference."""
    return frozendict(valmap_(pred, a))


def valmap(f: Callable[[V], W], d: Mapping[K, V]) -> Dict[K, W]:
    """Wrapper around `cytoolz.valmap`. Helps with type inference."""
    return valmap_(f, d)


def fd(a: Mapping[K, V]) -> Mapping[K, V]:
    """Makes a frozen dict. Needed for type inference."""
    return frozendict(a)


def fs(a: Collection[V]) -> FrozenSet[V]:
    """Makes a frozen set. Needed for type inference."""
    return frozenset(a)

from decimal import Decimal

from numpy.testing import assert_raises

from dg_commons import DgSampledSequence, seq_accumulate
from dg_commons.seq.seq_op import seq_differentiate, seq_integrate

ts = (1, 2, 3, 4, 5)
tsD = [Decimal(t) for t in ts]
val = [1, 2, 3, 4, 5]
seq = DgSampledSequence[float](ts, values=val)
seqD = DgSampledSequence[float](tsD, values=val)


def test_accumulate():
    expected = (1, 3, 6, 10, 15)
    seq_acc = seq_accumulate(seq)
    seqD_acc = seq_accumulate(seqD)
    assert seq_acc.values == expected
    assert seqD_acc.values == expected


def test_integrate():
    expected = (1.5, 4.0, 7.5, 12.0)
    seq_int = seq_integrate(seq)
    seqD_int = seq_integrate(seqD)
    assert seq_int.values == expected
    assert seqD_int.values == expected


def test_differentiate():
    expected = (1, 1, 1, 1)
    seq_dif = seq_differentiate(seq)
    seqD_dif = seq_differentiate(seqD)
    assert seq_dif.values == expected
    assert seqD_dif.values == expected


def test_illegal_assign():
    def _try_assign():
        seq.timestamps = [1, -3, 4.5]

    assert_raises(RuntimeError, _try_assign)

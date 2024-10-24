import numpy as np
from dg_commons import U, DgSampledSequence, iterate_with_dt
from dg_commons.seq.sequence import Timestamp


def get_max_jerk(commands: DgSampledSequence[U], t_range: tuple[Timestamp|None, Timestamp|None] = (None, None)):
    """
    Get the maximum jerk of the planned action sequence.
    Only timesteps within t_range are considered.
    """
    max_jerk = 0
    for it in iterate_with_dt(commands):
        if t_range[0] is not None and it.t1 < t_range[0]:
            continue
        if t_range[1] is not None and it.t1 > t_range[1]:
            break
        jerk = (it.v1.acc - it.v0.acc) / float(it.t1 - it.t0)
        if np.abs(jerk) > max_jerk:
            max_jerk = np.abs(jerk)
    return max_jerk


def get_acc_rms(commands: DgSampledSequence[U], t_range: tuple[Timestamp|None, Timestamp|None] = (None, None)):
    """
    comfort measurement according to ISO 2631. returns the rms of frequency weighted acceletation
    Only timesteps within t_range are considered.
    """
    t_range_min = commands.timestamps[0] if t_range[0] is None else t_range[0]
    t_range_max = commands.timestamps[-1] if t_range[1] is None else t_range[1]
    acc_time = np.array([command.acc for command, timestep in zip(commands.values, commands.timestamps) if timestep >= t_range_min and timestep <= t_range_max])
    st = 0.1
    acc_freqs = np.fft.rfft(acc_time)
    freqs = np.fft.rfftfreq(n=len(acc_time), d=st)
    acc_freq_weighted = [acc_freq * acc_freq_filter(freq) for acc_freq, freq in zip(acc_freqs, freqs)]
    acc_freq_weighted = np.array(acc_freq_weighted)
    acc_time_weighted = np.fft.irfft(acc_freq_weighted)
    acc_rms = 0
    for acc in acc_time_weighted:
        acc_rms += np.square(acc)
    acc_rms = np.sqrt(acc_rms/len(acc_time_weighted))
    return acc_rms


def acc_freq_filter(freq: float) -> float:
    """
    reference:
    Low order continuous-time filters for approximation of the ISO 2631-1 human vibration sensitivity weightings
    :param freq:
    :return: 3rd-order approximated weight for horizonal acceleration
    """
    w = (14.55 * freq ** 2 + 6.026 * freq + 7.725) / (freq ** 3 + 15.02 * freq ** 2 + 51.63 * freq + 47.61)
    return w

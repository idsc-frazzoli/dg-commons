import numpy as np
from dg_commons import U, DgSampledSequence


def get_max_jerk(commands: DgSampledSequence[U]):
    length = len(commands.timestamps)
    last_command = commands.values[0]
    last_t = commands.timestamps[0]
    max_jerk = 0
    for idx in range(1, length):
        cur_command = commands.values[idx]
        cur_t = commands.timestamps[idx]
        jerk = (cur_command.acc - last_command.acc) / float(cur_t - last_t)
        if np.abs(jerk) > max_jerk:
            max_jerk = np.abs(jerk)
        last_command = cur_command
        last_t = cur_t
    return max_jerk


def get_acc_rms(commands: DgSampledSequence[U]):
    """
    comfort measurement according to ISO 2631
    the rms of frequency weighted acceletation
    the
    :param states:
    :return:
    """
    acc_time = []
    for idx in range(len(commands.timestamps)):
        acc_time.append(commands.values[idx].acc)
    acc_time = np.array(acc_time)
    n_acc = len(acc_time)
    st = 0.1
    acc_freq = np.fft.rfft(acc_time)
    freq = np.fft.rfftfreq(n=n_acc, d=st)
    acc_freq_weighted = []
    for idx in range(len(freq)):
        acc_freq_weighted.append(acc_freq[idx] * acc_freq_filter(freq[idx]))
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

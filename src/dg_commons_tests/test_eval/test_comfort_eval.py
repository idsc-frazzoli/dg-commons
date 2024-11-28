import pickle
from dg_commons import PlayerName
from dg_commons.eval.comfort import get_max_jerk, get_acc_rms


def test_comfort_eval():
    file = open("src/dg_commons_tests/test_eval/logs/log.pickle", "rb")
    log = pickle.load(file)
    file.close()

    ego_name = PlayerName("Ego")
    ego_commands = log[ego_name].commands
    max_jerk = get_max_jerk(ego_commands)
    acc_rms = get_acc_rms(ego_commands)
    max_jerk_expected = 2.61
    acc_rms_expected = 0.13
    eps = 1e-2
    assert max_jerk_expected - eps <= max_jerk <= max_jerk_expected + eps
    assert acc_rms_expected - eps <= acc_rms <= acc_rms_expected + eps

import pickle
from dg_commons import PlayerName
from dg_commons.eval.comfort import get_max_jerk, get_acc_rms


def test_comfort_eval():
    file = open("src/dg_commons_tests/test_eval/logs/log.pickle", 'rb')
    log = pickle.load(file)
    file.close()

    ego_name = PlayerName("Ego")
    ego_commands = log[ego_name].commands
    max_jerk = get_max_jerk(ego_commands)
    acc_rms = get_acc_rms(ego_commands)
    print("Max jerk: " + str(max_jerk))
    print("Acc rms: " + str(acc_rms))


if __name__ == '__main__':
    test_comfort_eval()
    
from arg_utils import get_args
from recovery_rl.experiment import Experiment

if __name__ == '__main__':
    exp_cfg = get_args()
    experiment = Experiment(exp_cfg)
    experiment.run()

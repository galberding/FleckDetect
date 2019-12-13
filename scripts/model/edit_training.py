import argparse
import os
import stat
from yaml import load, dump, FullLoader

LEARNING_RATE = 0.0001
TRAINER = """#!/bin/bash\nsrun python trainer.py 2>&1 | tee "logs/{}.out"\n"""
SOLVER = "adam_solver.prototxt"
SOLVER_CONTENT = """train_net: "/media/compute/homes/galberding/FleckDetect/scripts/model/train.prototxt"
type: "Adam"
# lr for fine-tuning should be lower than when starting from scratch
#debug_info: true
test_net: [ "validation.prototxt" ]
test_iter: 54
test_interval: 1000;
test_compute_loss: true;
base_lr: {}
momentum: 0.9
momentum2: 0.999
lr_policy: "inv"
power: 0.3
gamma: 0.1
iter_size: 10
# stepsize should also be lower, as we're closer to being done
#stepsize: 7500
average_loss: 20
display: 20
max_iter: 100000
#momentum: 0.90
weight_decay: 0.0005
snapshot: 1000
snapshot_prefix: "snapshot/ras_{}"
# uncomment the following to default to CPU mode solving
# solver_mode: CPU
solver_mode: GPU
    """
def rename_train_cond(out_dir, name):

    config_path = os.path.join(out_dir, "config.yml")
    with open(config_path, "r") as f:
            conf = load(f, Loader=FullLoader)

    # print(SOLVER_CONTENT.format("blablub"))
    solver_path = os.path.join(out_dir, SOLVER)
    with open(solver_path, "w+") as f:
        f.write(SOLVER_CONTENT.format(conf["model"]["lr"], name))

    trainer_path = os.path.join(out_dir, "train.sh")
    with os.fdopen(os.open(trainer_path, os.O_RDWR | os.O_CREAT, 0o750), "w+") as t:
        t.write(TRAINER.format(name))
    conf["retrain"] = False

    with open(config_path, "w") as f:
        dump(conf, f)


# def prep_retrain():

if __name__ == "__main__":
    parser_ = argparse.ArgumentParser( \
        description='Controller Program to evaluate a trained model.', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser_.add_argument('name', type=str, default=None ,help='Set name for model and log')    
    args = parser_.parse_args()
    rename_train_cond("", args.name)
    print("Names adapted. Did you changed the paths to dataset yet?")

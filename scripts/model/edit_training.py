import argparse
import os
import stat
from yaml import load, dump, FullLoader

LEARNING_RATE = 0.0001
TRAINER = """#!/bin/bash\nsrun python trainer.py 2>&1 | tee "logs/{}.out"\n"""
SOLVER = "adam_solver.prototxt"
SOLVER_CONTENT = """train_net: [ "train.prototxt" ]
type: "Adam"
# lr for fine-tuning should be lower than when starting from scratch
#debug_info: true
test_net: [ "validation.prototxt" ]
test_iter: {}
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
    '''Use Model config to set train parameters and name the logs and snapshotted models'''
    config_path = os.path.join(out_dir, "config.yml")
    with open(config_path, "r") as f:
            conf = load(f, Loader=FullLoader)

    # print(SOLVER_CONTENT.format("blablub"))
    solver_path = os.path.join(out_dir, SOLVER)
    with open(solver_path, "w+") as f:
        f.write(SOLVER_CONTENT.format( conf["model"]["test_iter"], conf["model"]["lr"], name))

    trainer_path = os.path.join(out_dir, "train.sh")
    with os.fdopen(os.open(trainer_path, os.O_RDWR | os.O_CREAT, 0o750), "w+") as t:
        t.write(TRAINER.format(name))
    conf["retrain"] = False

    with open(config_path, "w") as f:
        dump(conf, f)


def set_net_proto_paths(root_dir, linkfile_path, proto_path):
    '''Set the "root" and "source" path in a prototxt network file.
    root_dir - path set as root
    linkfile_path - path set as source
    proto_path - path to prototxt file
    '''
    
    new_proto = []
    with open(proto_path, "r") as f:
        for line in f.readlines():
            if "root" in line:
                tmp = line.split(":")
                
                line = tmp[0]+ ': "' +root_dir + '"\n'
            if "source" in line:
                tmp = line.split(":")
                line = tmp[0]+ ': "' + linkfile_path + '"\n'
            new_proto.append(line)
    with open(proto_path, "w") as f:
        f.writelines(new_proto)

# def prep_retrain():

def set_model_config(model_config_path, solver_path=None, learn_rate=None, max_iter=None, test_iter=None, retrain=False, weight_path=None):
    '''Directly set the modelconfig patameter.
    If one parameter is not None it will be updated in the config file.
    ------------------------
    
    model_config_path - path to config
    solver - path to solver
    learn_rate -
    max_iter -
    test_iter - Parameters in the test set
    retrain - If false the baseweights are set otherwise the retrain weights are set
    weight_path - 
    '''
    with open(model_config_path, "r") as f: 
        conf = load(f, Loader=FullLoader)
        
    if solver_path:
        conf["solver"]["adam"] = solver_path
    
    if learn_rate:
        conf["model"]["lr"] = learn_rate

    if max_iter:
        conf["model"]["max_iter"] = max_iter 

    if test_iter:
        conf["model"]["test_iter"] = test_iter 

    if weight_path:
        conf["retrain"] = retrain
        if retrain:
            conf["weights"]["retrain"] = weight_path 
        else:
            conf["weights"]["base"] = weight_path 

    with open(model_config_path, "w") as f:
        dump(conf, f)




if __name__ == "__main__":
    # parser_ = argparse.ArgumentParser( \
    #     description='Controller Program to evaluate a trained model.', \
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser_.add_argument('name', type=str, default=None ,help='Set name for model and log')    
    # args = parser_.parse_args()
    # rename_train_cond("", args.name)
    # print("Names adapted. Did you changed the paths to dataset yet?")
    set_net_proto_paths("hey/ho/hey", "blub/linkfile.txt", "test.prototxt")

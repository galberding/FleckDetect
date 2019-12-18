from __future__ import division
import numpy as np
import sys
# caffe_root = '../.. /'
# sys.path.insert(0, caffe_root + 'python')
from yaml import load, dump, FullLoader
import caffe

def train(config):
    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    solver = caffe.AdamSolver(config["solver"]["adam"])
    if config["retrain"]:
        solver.net.copy_from(config["weights"]["retrain"])
    else:
        solver.net.copy_from(config["weights"]["base"])
    solver.step(config["model"]["max_iter"])
    

def load_config():
    with open("config.yml", "r") as f:
        return load(f, Loader=FullLoader)

if __name__ == "__main__":
    config = load_config()
    # print(config)
    train(config)

import subprocess as sp
import os
import numpy as np
import argparse
from tqdm import tqdm

def cal_metrics(pred_dir, gt_dir, out_path):
    '''Merge pred and gt dir and use the precompiled metric exe to calculate the
    corresponding values. The results will be written to the out_path'''
    preds = os.listdir(pred_dir)
    gts = os.listdir(gt_dir)
    pairs = []
    for p in preds: 
        p_tmp = p.split(".")[0]
        for gt in gts:
            if p_tmp == gt.split(".")[0]:
                 pairs.append((p, gt))
                 break
    print("Calculate metrics:")
    with open(out_path, "bw+") as out_file:
        for pred, gt in tqdm(pairs):
            gt_path = os.path.join(gt_dir, gt)
            pred_path = os.path.join(pred_dir, pred)
            exec_metrics(out_file, gt_path, pred_path)



def exec_metrics(fd, gt, pred):
    '''Calculate Metrics '''
    with sp.Popen(["./Metrics", pred, gt], stdout=sp.PIPE) as proc:
        fd.write(proc.stdout.read())

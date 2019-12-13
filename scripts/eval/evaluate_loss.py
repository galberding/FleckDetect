#!/usr/bin/env python

"""
Short script used to extract the minimal loss value 
and plot values from csv files.
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

def parse_loss_csv(filename):
    iterations = []
    losses = []
    min_loss = sys.maxsize
    min_index = -1
    index = 0
    with open(filename, 'r') as lossfile:
        first_line = True
        for line in lossfile:
            if first_line:
                first_line = False
                continue

            iterations.append(float(line[0:12].strip()))
            losses.append(float(line[12:25].strip()))
            if losses[index] < min_loss:
                min_loss = losses[index]
                min_index = index
            index += 1
    return np.array(iterations), np.array(losses), min_loss, min_index



def det_best_model(filelist, vis=False):
    for filename in filelist:
        loss_type = filename.replace('loss_', '').replace('.csv', '')

        iterations, losses, min_loss, min_index = parse_loss_csv(filename)
        if min_index > -1:
            print (loss_type, ': lowest loss', min_loss, 'at iteration', iterations[min_index])
            if vis:
                plt.plot(iterations, losses, '-', label=loss_type)
        else:
            print (loss_type, ': no lowest loss available!')

    if vis:
        plt.legend()
        plt.show()


def main():


    # initialize argument parser
    parser = argparse.ArgumentParser(
        description='Process a list of loss tables by evaluting the iteration with the lowest loss optionally visualizing them as graphs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add arguments
    parser.add_argument('filelist', nargs='+',
                        help='list of csv files to evaluate')
    parser.add_argument('-v', '--visual', action='store_true',
                        help='if the losses should be plotted')
    # parse arguments
    args = parser.parse_args()
    det_best_model(args.filelist, vis=args.visual)


if __name__ == "__main__":
    main()

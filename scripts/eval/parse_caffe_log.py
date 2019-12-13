#!/usr/bin/env python

import argparse
import re


# define how to format a list of values to print into a file
def format_list(list_to_format):
    return '{:<8}    {:<12}\n'.format(*list_to_format)


def parse_log(logf, train_out="loss_training.csv", val_out="loss_validation.csv"):
    pass
    number_regex = '(1.\d+e\+\d+|\d+)'
    # regex used to find training loss prints
    training_loss_regex = 'Iteration \d+, loss = ' + number_regex
    # regex used to find test iteration prints
    test_loss_iteration_regex = 'Iteration \d+, Testing net'
    # regex used to find test loss prints
    test_loss_regex = 'Test loss: ' + number_regex

    # open files into which the loss is printed
    loss_training = open(train_out, 'w')
    # loss_training = open(args.train_loss_file, 'w')
    loss_validation = open(val_out, 'w')
    # loss_validation = open(args.validation_loss_file, 'w')

    # print header lines
    loss_training.write(format_list(['Iteration', 'Loss']))
    loss_validation.write(format_list(['Iteration', 'Loss']))

    # save last test iteration because it is not printed in the same line as the loss
    test_iteration = -1
    # open logfile
    with open(logf, 'r') as logfile:
        # iterate lines in logfile
        for line in logfile:
            # search for training loss prin
            match = re.search(training_loss_regex, line) 
            if match:
                # found, so parse numbers, print them to file and continue
                numbers = re.findall(number_regex, match.group())
                loss_training.write(format_list(numbers))
                continue

            # search for test iteration print
            match = re.search(test_loss_iteration_regex, line)
            if match:
                # found, so parse iteration and continue
                test_iteration = re.search('\d+', match.group()).group()
                continue

            # search for test loss print
            match = re.search(test_loss_regex, line)
            if match:
                # found so print to file and continue
                loss_validation.write(format_list([test_iteration, re.search(number_regex, match.group()).group()]))

    # close files
    loss_training.close()
    loss_validation.close()



def main():


        # initialize argument parser
    parser = argparse.ArgumentParser(
        description='Process a logfile produced by caffe by creating loss tables for training and validation loss in csv files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add arguments
    parser.add_argument('logfile', type=str,
                        help='logfile created by a caffe training run')
    parser.add_argument('-t', '--train_loss_file', type=str, default='loss_training.csv',
                        help='file into which the training loss is writte')
    parser.add_argument('-v', '--validation_loss_file', type=str, default='loss_validation.csv',
                        help='file into which the validation loss is written')
    # parse arguments
    args = parser.parse_args()

    parse_log(args.logfile)


if __name__ == "__main__":
    main()

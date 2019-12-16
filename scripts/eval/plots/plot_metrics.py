import numpy as np
import matplotlib.pyplot as plt
# import sklearn.metrics as metrics
import sys
# from tabulate import tabulate
import os

def create_incremental_plot(metric_path, axs, name):
    '''Create PR- and ROC-Curve in two subplots.'''
    stats = np.loadtxt(open(metric_path, "r"), delimiter=',')
    mae = stats[0,0]
    width = 257
    delim = 1
    precisions = stats[:, delim:delim+width]
    delim += width
    recalls = stats[:, delim:delim+width]
    delim += width
    fbetas = stats[:, delim:delim+width]
    delim += width
    fprs = stats[:, delim:delim+width]
    delim += width
    tprs = stats[:, delim:delim+width]
    mean_precison = (precisions.mean(axis=0))
    mean_recall = recalls.mean(axis=0)
    # Exclude first and last point for better plot results
    mean_precison = mean_precison[1:-1]
    mean_recall = mean_recall[1:-1]
    mean_fpr = fprs.mean(axis=0)
    mean_tpr = tprs.mean(axis=0)
    axs[0].set_title("PR-Curve")   
    axs[0].set_xlabel("Recall")
    axs[0].set_ylabel("Precision")
    axs[0].grid(True)
    axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[0].plot(mean_recall, mean_precison, label=name)

    axs[1].plot(mean_fpr, mean_tpr, label=name)   
    axs[1].set_title("ROC-Curve")
    axs[1].set_xlabel("Fpr")
    axs[1].set_ylabel("Tpr")
    axs[1].grid(True)
    axs[1].plot([0,1], [0,1], "--", c="#000000")
    axs[1].set_xlim([0, 1])
    axs[1].set_ylim([0, 1])

    # axs[0].scatter(mean_recall, mean_precison,s=10, c="#29F041")



    # Add stats to axis

def create_plots(metric_path, out_dir):
    '''First parse csv data from metric_path, subsequent write plots to out_dir'''
    stats = np.loadtxt(open(metric_path, "r"), delimiter=',')
    mae = stats[0,0]
    width = 257
    delim = 1
    precisions = stats[:, delim:delim+width]
    delim += width
    recalls = stats[:, delim:delim+width]
    delim += width
    fbetas = stats[:, delim:delim+width]
    delim += width
    fprs = stats[:, delim:delim+width]
    delim += width
    tprs = stats[:, delim:delim+width]
    mean_precison = (precisions.mean(axis=0))
    mean_recall = recalls.mean(axis=0)
    mean_fpr = fprs.mean(axis=0)
    mean_tpr = tprs.mean(axis=0)
    # print(fprs.shape)
    # with open("stats.csv", "w+") as f:
    # for i in range(257):
    #     print("|{}|{}|{}|{}|{}|".format(i-1, mean_fpr[i], mean_tpr[i], mean_precison[i], mean_recall[i]))
    roc_path = os.path.join(out_dir, "roc_curve.pdf")
    pr_path = os.path.join(out_dir, "pr_curve.pdf")
    plot_roc_curve(mean_fpr, mean_tpr, roc_path)
    plt.close()
    plot_pr_curve(mean_precison, mean_recall, pr_path)
    plt.close()

# def createTable():
#     res = [["precision"], ["recall"], ["tpr"],
#            ["fpr"], ["fbeta"], ["mae"], ["AuC"]]
#     header = ["Metric"]
#     for path in sys.argv[1:]:
#         stats = (plot_stats(path))
#         for i in range(len(res)):
#             res[i].append(stats[i])
#         header.append(path)

#     print(tabulate(res, headers=header, tablefmt='github'))
    

def plot_pr_curve(mean_precison, mean_recall, out_path):
    mean_precison = mean_precison[1:-1]
    mean_recall = mean_recall[1:-1]
    plt.title("PR-Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot(mean_recall, mean_precison)
    plt.scatter(mean_recall, mean_precison,s=10, c="#29F041")
    # plt.show()
    plt.savefig(out_path)
    # auc = metrics.auc(mean_recall, mean_precison)
    print("PR-Curve Auc: ", auc)

def plot_roc_curve(mean_fpr, mean_tpr, out_path):
    plt.plot(mean_fpr, mean_tpr )
    plt.scatter(mean_fpr, mean_tpr, s=10, c="#29F041")
    # plt.scatter([1], [1] )
    plt.title("ROC-Curve")
    plt.xlabel("Fpr")
    plt.ylabel("Tpr")
    plt.grid()
    plt.plot([0,1], [0,1], "--", c="#000000")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(out_path)
    # auc = metrics.auc(mean_fpr, mean_tpr)
    print("ROC-Curve Auc: ", auc)


def plot_thresh(path):
    stats = np.loadtxt(open(path, "rb"), delimiter=',')
    print(stats.shape)
    for i in range(257):
        th1_fpr = stats[0+i, :]
        th1_tpr = stats[257+i, :]
        plt.plot(th1_fpr, th1_tpr)
        plt.show()
    

   

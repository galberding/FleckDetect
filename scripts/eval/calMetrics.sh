#!/bin/bash

#######################################################
#######################################################
# Wrapper to calculate metrics for Saliency maps (predictions) and corresponding ground truth.
# DATA_PAIRS  - points to a file which contains the ground truth and corresponding prediction in each line (gt, pred) in a csv file.
# PRED_OFFSET - Path to predictions corresponding to the pred in DATA_PAIRS
# GT_OFFSET - Path to ground truth corresponding to the pred in DATA_PAIRS
# Result: A file metric_results.csv will be created which contains:
#       precision, recall, truePositiveRate, falsePositiveRate, fBeta, mae
#
# Note: Both prediction and gt are .png with similar name except that pred ends with suffix _pred.png.
#######################################################
#######################################################

PATH_COUNT=0
OPTION_ENABLE_RAW_CALCULATION=0
OPTION_EVAL_METRICS=0

while getopts  "abc:def:ghio:" flag
do
  echo "$flag" $OPTIND $OPTARG
    # var=$((var+1))
    # echo "Var: $var"

    case "$flag" in 
    a)
    echo "Enable complete pipeline"
    OPTION_ENABLE_RAW_CALCULATION=1
    OPTION_EVAL_METRICS=1
    ;;
    c) 
        echo "Hello c $OPTARG"
    ;;
    o) 
        echo "Option $OPTARG"
        case  "$OPTARG" in 
            "ERC")
                echo "Recalculationg Metrics!"
                OPTION_ENABLE_RAW_CALCULATION=1
                ;;
            "EM")
                echo "Eval Metrics!"
                OPTION_EVAL_METRICS=1
        esac
        ;;
    h)
    echo "<--Help-->"
    echo "-a  Enable Raw metric calculation and evaluation"
    echo "-o <option> "
    echo "  ERC: Enable Raw calculation"
    echo "  EM : Eval Metrics"
    ;;

    esac
done



#Replace with path to predictions
# PRED_OFFSET=/home/schorschi/Documents/SS19/FleckDetect/Results/predictions_11400_val_adam

PRED_OFFSET=/home/schorschi/Documents/SS19/FleckDetect/Results/msra_auth_preds
# Author results
# PRED_OFFSET=/home/schorschi/Documents/SS19/FleckDetect/Results/val_auth
# Replace with path to ground truth
# GT_OFFSET=/home/schorschi/Documents/SS19/FleckDetect/Results/gt
GT_OFFSET=/home/schorschi/Documents/SS19/FleckDetect/MSRA-B/gt
# Assumes all files are listed in a csv file: gt, pred ...
DATA_PAIRS=/home/schorschi/Documents/SS19/FleckDetect/Results/msrab_data.csv

# OUT_BIN_1=stats/current.csv
# OUT_BIN_2=stats/current_msrab.csv
OUT_BIN_2=stats/auth_msrab_all.csv
# OUT_BIN_1=stats/metric_auth_results_bin_mean.csv
# OUT_BIN_2=stats/metric_auth_results_bin_fbeta.csv
if [ $OPTION_ENABLE_RAW_CALCULATION = 1 ] 
then
    #clear metric_results.csv
    # echo -n > "$OUT_BIN_1"
    echo -n > "$OUT_BIN_2"

    cat $DATA_PAIRS| while IFS=, read -r gt pred
    do
    echo "Calc metrics for: ${gt}"
    # echo
    #  prediction=`echo -n PRED_OFFSET/$pred`
    #  ground=`echo -n GT_OFFSET/$gt`
    # ./Metrics "${FILE_OFFSET}/${pred}" "${FILE_OFFSET}/${gt}";
    #  ./Metrics $PRED_OFFSET/$pred $GT_OFFSET/$gt >> metric_results.csv
    # Ugly fix because predictions are not named preperly


    # ./Metrics $PRED_OFFSET/$pred $GT_OFFSET/$gt 0 >> "$OUT_BIN_1"
    ./Metrics $PRED_OFFSET/$pred $GT_OFFSET/$gt 1 >> "$OUT_BIN_2"
    #   ./Metrics $PRED_OFFSET/$gt $GT_OFFSET/$gt 0 >> "$OUT_BIN_1"
    #   ./Metrics $PRED_OFFSET/$gt $GT_OFFSET/$gt 1 >> "$OUT_BIN_2"
    
    done
fi
# for file in `ls *.png`; do newfile=`echo $file | sed 's/.png/_pred.png/'`; mv $file $newfile; done

if [ $OPTION_EVAL_METRICS = 1 ] 
then
    python eval_metrics.py ${OUT_BIN_2}
fi
#!/bin/bash
# WORKSPACE=combined_on_fleck

# for WORKSPACE in "combined_on_fleck" "msrab"
# do
#     TEST=Blu
#     COMMAND="
#     echo $TEST
#     source bash_sub
#     source envrc_fleck_detect
#     cd FleckDetect/scripts
#     for VAR in 20000 40000 60000 80000
#     do 
#         echo $VAR
#         echo srun python controller.py -w  ${WORKSPACE} --sel \$VAR --seg-all
#     done
#     "
#     echo $COMMAND
# ssh -tt gpu << EOF
# $COMMAND
# exit
# exit
# EOF

#     echo "Start calculating Metrics"
#     for VAR in 20000 40000 60000 80000
#     do 
#         echo $VAR
#         echo "srun python controller.py -w  $WORKSPACE --sel $VAR --seg-all"
#     done

#     echo "python controller.py -w  $WORKSPACE --plot-models"
# done 



# for WORKSPACE in "combined_on_fleck" "msrab" "soc" "tamino_on_fleck" "combined" "auth_on_soc"
for WORKSPACE in "msrab_on_fleck" "soc_on_fleck"
do 
    for VAR in 10000 20000 40000 60000 # 80000 # model iterations to test
    do 
        echo $VAR
        # srun python controller.py -w ${WORKSPACE} --sel $VAR --seg-all
        python controller.py -w ${WORKSPACE} --sel $VAR --cal-all-metrics
    done
    python controller.py -w ${WORKSPACE} --plot-models
done 

mail -s "Job Finished!" "galberding@techfak.uni-bielefeld.de" << EOF
Done!
EOF
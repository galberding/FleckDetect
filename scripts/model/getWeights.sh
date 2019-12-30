#!/bin/bash
module use /vol/ai/share/modulefiles 
module add sod_model/ras 
cp  $MODEL_ZOO/RAS/weights.caffemodel .
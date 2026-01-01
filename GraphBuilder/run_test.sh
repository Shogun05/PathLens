#!/bin/bash
# Run Sat2Graph inference on test set with pretrained model

cd "$(dirname "$0")/model"
source ../venv37/bin/activate
python train.py \
    -model_save ../output/test_results \
    -instance_id baseline \
    -image_size 352 \
    -model_recover ../data/20citiesModel/model \
    -mode test 2>&1 | tee ../inference.log

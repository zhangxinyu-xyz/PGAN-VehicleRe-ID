#!/bin/bash

##### test ####
python tools/test.py --config_file='configs/softmax_triplet_veriwild.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('../PGAN_models/VERI-WILD/model_best.pth')" \
OUTPUT_DIR "('../PGAN_models/VERI-WILD/')"







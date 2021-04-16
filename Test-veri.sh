#!/bin/bash

##### test ####
python tools/test.py --config_file='configs/softmax_triplet_veri.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('../PGAN_models/VeRi/model_best.pth')" \
OUTPUT_DIR "('../PGAN_models/VeRi/')"







#!/bin/bash

##### test ####
python tools/test.py --config_file='configs/softmax_triplet_vehicleid.yml' \
MODEL.DEVICE_ID "('0')" \
MODEL.PRETRAIN_CHOICE "('self')" \
TEST.WEIGHT "('../PGAN_models/PKU-VehicleID/model_best.pth')" \
OUTPUT_DIR "('../PGAN_models/PKU-VehicleID/')"





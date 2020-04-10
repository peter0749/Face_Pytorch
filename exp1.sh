#!/bin/bash

TRAIN_ROOT=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/CASIA-WebFace
TRAIN_FILE_LIST=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/casia-webface-list.txt
LFW_TEST_ROOT=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/lfw/lfwa
LFW_FILE_LIST=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/lfw/pairs.txt
APD_TEST_ROOT=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/APD/C
APD_POSITIVE=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/APD/positive_pairs.txt
APD_NEGATIVE=/media/peter0749/63b135e6-bbe2-4e5b-8456-72b5608b7814/APD/negative_pairs.txt
LOSS_TYPE="Softmax"
DIM=512
BS=64
SAVE_FREQ=500
TEST_FREQ=500
BACKBONE="Res50_IR"
PREFIX="Res50_IR_Softmax"
SAVE_DIR="./model_Res50"

python train.py --train_root "$TRAIN_ROOT" --train_file_list "$TRAIN_FILE_LIST" --lfw_test_root "$LFW_TEST_ROOT" --lfw_file_list "$LFW_FILE_LIST" --apd_test_root "$APD_TEST_ROOT" --apd_positive_pair "$APD_POSITIVE" --apd_negative_pair "$APD_NEGATIVE" --margin_type "$LOSS_TYPE" --feature_dim $DIM --batch_size $BS --save_freq $SAVE_FREQ --test_freq $TEST_FREQ --gpus 0 --backbone "$BACKBONE" --model_pre "$PREFIX" --save_dir "$SAVE_DIR"


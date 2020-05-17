#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

$PYTHON new_ae_train.py --dump-root  dump/2017/ --preset ae_hp.json --checkpoint-dir ae_exp/0305AE_0/ \
                --log-event-path ae_tensorboard/0305AE_0/ --feat=mfcc --use-norm  #--checkpoint=ae_exp/0303AE_0/checkpoint_step000135000.pth

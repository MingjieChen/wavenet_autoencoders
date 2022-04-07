#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

$PYTHON train.py --dump-root  dump/2017/ --preset hp.json --checkpoint-dir exp/0224MFCC_0/ \
                --log-event-path tensorboard/0224MFCC_0/ --feat=mfcc --use-norm

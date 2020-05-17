#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON wavenet_catae_train.py --dump-root  dump/2019/english/  --preset exp/0402catae_0/hparams.json --checkpoint-dir exp/0402catae_0/ \
#                --log-event-path tensorboard/0402catae_0/ --feat=mfcc --use-norm  --checkpoint=exp/0402catae_0/checkpoint_step000240000.pth
exp=0422catae_0
step=checkpoint_latest.pth
hp=catae_hp.json
load_hp=exp/$exp/hparams.json

$PYTHON wavenet_catae_train.py --dump-root  dump/2019/english/ --preset $load_hp --checkpoint-dir exp/$exp/ \
                --log-event-path tensorboard/$exp/ --feat=mfcc --use-norm  --checkpoint=exp/$exp/$step

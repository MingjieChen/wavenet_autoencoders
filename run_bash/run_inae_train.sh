#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON wavenet_ae_train.py --dump-root  dump/2017/ --preset wvae_hp.json --checkpoint-dir exp/0321wvae_0/ \
#                --log-event-path tensorboard/0321wvae_0/ --feat=mfcc --use-norm   #--checkpoint=exp/0321wvae_0/checkpoint_step000016000.pth


exp=$1
#step=checkpoint_step000090000.pth
step=checkpoint_latest.pth
$load_hp=exp/$exp/hparams.json
#hp=inae_hp.json
hp=$2
#lan=surprise
lan=$3


$PYTHON wavenet_inae_train.py --dump-root  dump/2019/$lan/ --preset $load_hp  --checkpoint-dir exp/$exp/ \
                --log-event-path tensorboard/$exp/ --feat=mfcc --use-norm  --checkpoint=exp/$exp/$step

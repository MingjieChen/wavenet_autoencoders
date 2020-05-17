#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON wavenet_ae_train.py --dump-root  dump/2017/ --preset wvae_hp.json --checkpoint-dir exp/0321wvae_0/ \
#                --log-event-path tensorboard/0321wvae_0/ --feat=mfcc --use-norm   #--checkpoint=exp/0321wvae_0/checkpoint_step000016000.pth
$PYTHON wavenet_ae_train.py --dump-root  dump/2019/english/ --preset exp/0323wvae_0/hparams.json --checkpoint-dir exp/0323wvae_0/ \
                --log-event-path tensorboard/0323wvae_0/ --feat=mfcc --use-norm  --checkpoint=exp/0323wvae_0/checkpoint_step000160000.pth

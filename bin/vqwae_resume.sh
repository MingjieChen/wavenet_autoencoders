#!/bin/bash
source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/bin/activate torch_0.4

#$PYTHON wavenet_vqvae_train.py --dump-root  dump/2017/ --preset wv_vqvae_hp.json --checkpoint-dir exp/0321vqvae_0/ \
#                --log-event-path tensorboard/0321vqvae_0/ --feat=mfcc --use-norm   --checkpoint=exp/0321vqvae_0/checkpoint_step000264000.pth



exp=2022_0407_batch40
#step=checkpoint_step000200000.pth
step=checkpoint_latest.pth
#hp=wv_vqvae_hp.json
hp=hps/vqwae.json
#load_hp=exp/$exp/hparams.json
lan=english

python vqwae_train.py --dump-root  dump/2019/$lan/ --preset $hp --checkpoint-dir exp/$exp/ \
                --log-event-path tensorboard/$exp/ --feat=mfcc --use-norm  --checkpoint=exp/$exp/checkpoint_latest.pth

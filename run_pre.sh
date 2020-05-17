#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON train.py --dump-root  dump/lj/logmelspectrogram/norm/ --preset hp.json --checkpoint-dir exp/0108LJ/ \
#                --log-event-path tensorboard/0108LJ/

#$PYTHON  mksubset_2019.py  2020/2019/  dump/2019/ scp/2019/english/
#$PYTHON  mksubset_2019.py  2020/2019/  dump/2019/ scp/2019/surprise/



#$PYTHON preprocess_2017.py  scp/2017/dev_src_dst.json dump/2017/dev/ \
#    2017_speaker2ind.json --preset=hp.json
#$PYTHON preprocess_2019.py  scp/2019/english/test_src_dst.json dump/2019/english/test/ \
#    2019_speaker2ind.json --preset=hp.json

#$PYTHON preprocess_2019.py  scp/2019/surprise/test_src_dst.json dump/2019/surprise/test/ \
#    2019_speaker2ind_surprise.json --preset=hp.json

#$PYTHON compute_mean_var.py scp/2017/test_src_dst.json dump/2017/test/mvn_mfcc.joblib  mfcc --verbose=1
#$PYTHON compute_mean_var.py scp/2019/english/test_src_dst.json dump/2019/english/test/mvn_mfcc.joblib  mfcc --verbose=1
#$PYTHON compute_mean_var.py scp/2019/surprise/train_src_dst.json dump/2019/surprise/train_no_dev/mvn_mfcc.joblib  mfcc --verbose=1

#$PYTHON normalize.py  scp/2017/test_src_dst.json mfcc dump/2017/test/mvn_mfcc.joblib 



$PYTHON normalize.py  scp/2019/surprise/test_src_dst.json mfcc dump/2019/surprise/train_no_dev/mvn_mfcc.joblib 

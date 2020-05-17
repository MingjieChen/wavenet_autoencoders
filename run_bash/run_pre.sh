#!/bin/bash
source activate torch_0.4
PYTHON=/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/torch_0.4/bin/python

#$PYTHON train.py --dump-root  dump/lj/logmelspectrogram/norm/ --preset hp.json --checkpoint-dir exp/0108LJ/ \
#                --log-event-path tensorboard/0108LJ/

data_root=2020/2019/
dump_dir=dump/2019/
scp_dir=scp/2019/

echo "start preprocessing for data root ${data_root} dump_dir ${dump_dir} scp_dir ${scp_dir}"
echo "stage 1: mksubset"
for lan in english surprise; do
    $PYTHON  mksubset_2019.py  $data_root  $dump_dir ${scp_dir}$lan/
done
echo "stage 1 done"
#$PYTHON  mksubset_2019.py  2020/2019/  dump/2019/ scp/2019/surprise/



#$PYTHON preprocess_2017.py  scp/2017/dev_src_dst.json dump/2017/dev/ \
#    2017_speaker2ind.json --preset=hp.json

echo "stage 2 extract features"
for lan in english surprise; do
    for sub in train_no_dev dev test; do 
        $PYTHON preprocess_2019.py  ${scp_dir}$lan/${sub}_src_dst.json ${dump_dir}$lan/$sub/ \
            2019_speaker2ind.json --preset=hp.json
    done
done
echo "stage 2 done"
#$PYTHON preprocess_2019.py  scp/2019/surprise/test_src_dst.json dump/2019/surprise/test/ \
#    2019_speaker2ind_surprise.json --preset=hp.json

#$PYTHON compute_mean_var.py scp/2017/test_src_dst.json dump/2017/test/mvn_mfcc.joblib  mfcc --verbose=1

echo "stage 3 compute mvn"
for lan in english surprise; do 
    $PYTHON compute_mean_var.py ${scp_dir}${lan}/train_src_dst.json ${dump_dir}${lan}/train_no_dev/mvn_mfcc.joblib  mfcc --verbose=1
done
echo "stage 3 done"
#$PYTHON compute_mean_var.py scp/2019/surprise/train_src_dst.json dump/2019/surprise/train_no_dev/mvn_mfcc.joblib  mfcc --verbose=1

echo "stage 4 normlaize"
for lan in english surprise; do
    for sub in train_no_dev dev test; do
        $PYTHON normalize.py  ${scp_dir}/${lan}/${sub}_src_dst.json mfcc ${sump_dir}${lan}/train/mvn_mfcc.joblib 
    done
done
echo "stage 4 done"

#$PYTHON normalize.py  scp/2019/surprise/test_src_dst.json mfcc dump/2019/surprise/train_no_dev/mvn_mfcc.joblib 

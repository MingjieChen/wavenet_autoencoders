#!/bin/bash
source /share/mini1/sw/std/python/anaconda3-2019.07/v3.7/bin/activate torch_0.4

stage=1
data_root=/share/mini1/res/t/repr/com/unsup-en/zsc2020/2020/2019
dump_dir=dump/2019/
scp_dir=scp/2019/
feat_type=mfcc
languages=("english")
echo "start preprocessing for data root ${data_root} dump_dir ${dump_dir} scp_dir ${scp_dir}"

if [[ $stage -le 1 ]]; then
    echo "stage 1: mksubset"
    for lan in $languages ; do
        python  mksubset_2019.py $lan  $data_root  $dump_dir ${scp_dir}/$lan/
    done
    echo "stage 1 done"
fi    

if [[ $stage -le 2 ]]; then
    echo "stage 2 extract features"
    for lan in $languages ; do
        for sub in train_no_dev dev test; do 
            python preprocess_2019.py  ${scp_dir}/$lan/${sub}_src_dst.json ${dump_dir}/$lan/$sub/ \
                2019_speaker2ind_${lan}.json --preset=hps/hp.json
        done
    done
    echo "stage 2 done"
fi

if [[ $stage -le 3 ]]; then
    echo "stage 3 compute mvn"
    for lan in $languages ; do 
        python compute_mean_var.py ${scp_dir}${lan}/train_src_dst.json ${dump_dir}${lan}/train_no_dev/mvn_mfcc.joblib  $feat_type --verbose=1
    done
    echo "stage 3 done"
fi    

if [[ $stage -le 4 ]]; then
    echo "stage 4 normlaize"
    for lan in $languages ; do
        for sub in train_no_dev dev test; do
            python normalize.py  ${scp_dir}/${lan}/${sub}_src_dst.json ${feat_type} ${dump_dir}/${lan}/train_no_dev/mvn_${feat_type}.joblib 
        done
    done
    echo "stage 4 done"
fi    


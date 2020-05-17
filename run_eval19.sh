#!/bin/bash
source activate zerospeech2020_1
export ZEROSPEECH2020_DATASET=./2020/
export  HDF5_USE_FILE_LOCKING=FALSE



infer_dir=exp/0511vqvae_0/infer_350k/
result_dir=exp/0511vqvae_0/res_350k/
lan=english

if [[ ! -e $result_dir ]]; then 
    mkdir -p $result_dir
fi

echo "infer_dir ${infer_dir} result_dir ${result_dir}"

#for lan in mandarin english french;
#do
/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-evaluate 2019  -j10 ${infer_dir}   -o ${result_dir}${lan}.json
#    echo "eval for $lan to ${result_dir}${lan}.json"
#done




#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-validate ae_exp/0223Mel_0/inference/



#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-evaluate 2017-track1 -j10 exp/0227VQVAE_0/152000inference/   -o exp/0227VQVAE_0/152kresult.json
#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-evaluate 2017-track1 -j10 ae_exp/0223Mel_0/inference/  -o ae_exp/0223Mel_0/result.json

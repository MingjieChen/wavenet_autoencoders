#!/bin/bash
source activate zerospeech2020_1
export ZEROSPEECH2020_DATASET=./2020/
export  HDF5_USE_FILE_LOCKING=FALSE



/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-validate $1 --njobs 10




#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-validate ae_exp/0223Mel_0/inference/



#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-evaluate 2017-track1 -j10 exp/0227VQVAE_0/152000inference/   -o exp/0227VQVAE_0/152kresult.json
#/share/mini1/sw/std/python/anaconda3-2019.07/v3.7/envs/zerospeech2020_1/bin/zerospeech2020-evaluate 2017-track1 -j10 ae_exp/0223Mel_0/inference/  -o ae_exp/0223Mel_0/result.json

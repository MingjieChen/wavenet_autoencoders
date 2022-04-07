#!/bin/bash

job=bin/run_wv_vqvae_train.sh
runs=20
ngpu=4
slots=8

#jid=$(submitjob -g${ngpu} -m 10000 -M${slots}  -o -l gputype="GeForceGTXTITANX|GeForceGTX1080Ti" -eo logs/2022_0405_batch.log  $job | grep -E [0-9]+ )
jid=$1
for ((n=1;n<$runs;n++));do
    echo "$n auto resume"
    jid=$(submitjob  -g${ngpu}  -m 10000  -M${slots}  -w $jid   -o -l gputype="GeForceGTXTITANX|GeForceGTX1080Ti"  -eo  logs/2022_0405_batch.log  $job | grep -E [0-9]+ )
done


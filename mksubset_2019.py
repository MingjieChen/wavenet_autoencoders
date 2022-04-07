# coding: utf-8
"""
Make subset of dataset

usage: mksubset_2019.py [options] <language> <in_dir> <out_dir> <scp_dir> 

options:
    -h, --help               Show help message.
"""
from docopt import docopt
import librosa
from glob import glob
from os.path import join, basename, exists, splitext
from tqdm import tqdm
import sys
import os
from shutil import copy2
from scipy.io import wavfile
import numpy as np
import  json



def read_wav(src_file):
    sr,x = wavfile.read(src_file)
    return sr,x
def write_wav(dst_path,sr,x):
    wavefile.write(dst_path,sr,x)



if __name__ == "__main__":
    args = docopt(__doc__)
    in_dir = args["<in_dir>"] # e.g.: 2020/2019/
    out_dir = args["<out_dir>"] # e.g.: dump/2019/
    scp_dir = args['<scp_dir>'] # e.g.: scp/2019/
    LAN = args['<language>']
    signed_int16_max = 2**15
    tr_src_files = []
    dev_src_files = []
    test_src_files = []
    tr_dev_src_fs = sorted(glob(in_dir+f'/{LAN}/train/unit/*.wav')) + sorted(glob(in_dir + f"/{LAN}/train/voice/*.wav"))
    te_src_fs = sorted(glob(in_dir + f'/{LAN}/test/*.wav'))
    num = len(tr_dev_src_fs)
    dev_num = int(0.01 * num)
    tr_src_files.extend(tr_dev_src_fs[dev_num:])
    dev_src_files.extend(tr_dev_src_fs[:dev_num])
    test_src_files.extend(te_src_fs)
    print(f"total number of train utts {len(tr_src_files)} dev utts {len(dev_src_files)} test {len(test_src_files)}",flush=True)

    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()   

    os.makedirs(out_dir + 'train_no_dev/',exist_ok=True)
    os.makedirs(out_dir + 'dev/',exist_ok=True)
    os.makedirs(out_dir + 'test/',exist_ok=True)

    os.makedirs(scp_dir,exist_ok=True)
    

    speakers = [] 
    tr_src_dst_files = []
    tr_src_dst_f = open(f'{scp_dir}/train_no_dev_src_dst.json','w')
    for src_path in tqdm(tr_src_files):
        sr,x = read_wav(src_path)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / signed_int16_max
        scaler.partial_fit(x.astype(np.float64).reshape(-1,1))
        lan = src_path.split('/')[-4]
        fname = src_path.split('/')[-1]
        sp_fid = fname.split('.')[0]
        sp = sp_fid.split('_')[0]
        if sp not in speakers:
            speakers.append(sp)
        dst_path = out_dir + f'/{lan}/train_no_dev/{sp_fid}/'
        os.makedirs(dst_path,exist_ok=True)
        tr_src_dst_files.append( (src_path,dst_path))
    json.dump(tr_src_dst_files,tr_src_dst_f)


    dev_src_dst_files = []
    dev_src_dst_f = open(f'{scp_dir}/dev_src_dst.json','w')
    for src_path in tqdm(dev_src_files):
        sr,x = read_wav(src_path)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / signed_int16_max
        scaler.partial_fit(x.astype(np.float64).reshape(-1,1))
        lan = src_path.split('/')[-4]
        fname = src_path.split('/')[-1]
        sp_fid = fname.split('.')[0]
        sp = sp_fid.split('_')[0]
        if sp not in speakers:
            speakers.append(sp)
        dst_path = out_dir + f'/{lan}/dev/{sp_fid}/'
        os.makedirs(dst_path,exist_ok=True)
        dev_src_dst_files.append( (src_path,dst_path) )
    json.dump(dev_src_dst_files,dev_src_dst_f)

    test_src_dst_files = []
    test_src_dst_f = open(f'{scp_dir}/test_src_dst.json','w')
    for src_path in tqdm(test_src_files):
        sr,x = read_wav(src_path)
        if x.dtype == np.int16:
            x = x.astype(np.float32) / signed_int16_max
        lan = src_path.split('/')[-3]
        fname = src_path.split('/')[-1]
        sp_fid = fname.split('.')[0]
        dst_path = out_dir + f'/{lan}/test/{sp_fid}/'
        os.makedirs(dst_path,exist_ok=True)
        test_src_dst_files.append( (src_path,dst_path) )
    json.dump(test_src_dst_files,test_src_dst_f)


    speaker2ind = {sp:ind for ind,sp in enumerate(speakers)}
    f_sp = open(f'2019_speaker2ind_{LAN}.json','w')
    json.dump(speaker2ind,f_sp)
    print("Waveform min: {}".format(scaler.data_min_))
    print("Waveform max: {}".format(scaler.data_max_))
    absmax = max(np.abs(scaler.data_min_[0]), np.abs(scaler.data_max_[0]))
    print("Waveform absolute max: {}".format(absmax))
    if absmax > 1.0:
        print("There were clipping(s) in your dataset.")
    print("Global scaling factor would be around {}".format(1.0 / absmax))


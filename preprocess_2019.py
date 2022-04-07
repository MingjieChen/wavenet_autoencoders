# coding: utf-8
"""
Preprocess dataset

usage: preprocess.py [options] <scp_dir> <out_dir> <sp2ind_dir>

options:
    --hparams=<parmas>       Hyper parameters [default: ].
    --preset=<json>          Path of preset parameters (json).
    -h, --help               Show help message.
"""

from functools import partial
import numpy as np
import os
import sys
import json
from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists, basename, splitext
import audio
import librosa
from glob import glob
from os.path import join
import json
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw
from docopt import docopt
from tqdm import tqdm
def preprocess( in_dir, out_root,sp2ind_dir):
        #os.makedirs(out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, out_dir,sp2ind_dir )
    write_metadata(metadata, out_dir)
def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[1] for m in metadata])
    sr = hparams.sample_rate
    hours = frames / 100 / 3600
    print('Wrote %d utterances, %d time steps (%.2f hours)' % (len(metadata), frames, hours))
    print('Min frame length: %d' % min(m[1] for m in metadata))
    print('Max frame length: %d' % max(m[1] for m in metadata))
def build_from_path(in_dir,out_dir,sp2ind_dir):
    metadata = []
    src_f = open(in_dir)
    src_files = json.load(src_f)
    sp_f = open(sp2ind_dir,'r')
    sp2ind = json.load(sp_f)
    for wav_path, dst_path in tqdm(src_files):
        _data = _process_utterance(dst_path,wav_path,sp2ind,'dummy')
        metadata.append(_data)
    return metadata


def _process_utterance(out_dir,wav_path,sp2ind,text):
    
    sp = wav_path.split('/')[-1].split('.')[0].split('_')[0]
    if sp in sp2ind:
        sp_ind = sp2ind[sp]
    else:
        sp_ind = -1
       
    wav = audio.load_wav(wav_path)
    if not 'test' in wav_path:
        wav,_ = librosa.effects.trim(wav,top_db=60,frame_length=2048,hop_length=512)

    if hparams.highpass_cutoff > 0.0:
        wav = audio.low_cut_filter(wav, hparams.sample_rate, hparams.highpass_cutoff)

    if is_mulaw_quantize(hparams.input_type):
        # Trim silences in mul-aw quantized domain
        silence_threshold = 0
        if silence_threshold > 0:
            # [0, quantize_channels)
            out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
            start, end = audio.start_and_end_indices(out, silence_threshold)
            wav = wav[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        constant_values = P.mulaw(0.0, hparams.quantize_channels - 1)
        out_dtype = np.float32
    else:
        # [-1, 1]
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.logmelspectrogram(wav).astype(np.float32).T
    mfcc = audio.mfcc(wav).astype(np.float32).T
    if hparams.global_gain_scale > 0:
        wav *= hparams.global_gain_scale

    # Time domain preprocessing
    if hparams.preprocess is not None and hparams.preprocess not in ["", "none"]:
        f = getattr(audio, hparams.preprocess)
        wav = f(wav)

    # Clip
    if np.abs(wav).max() > 1.0:
        print("""Warning: abs max value exceeds 1.0: {}""".format(np.abs(wav).max()))
        # ignore this sample
        #return ("dummy", "dummy","dummy", -1,-1, "dummy")

    wav = np.clip(wav, -1.0, 1.0)

    # Set waveform target (out)
    if is_mulaw_quantize(hparams.input_type):
        out = P.mulaw_quantize(wav, hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        out = P.mulaw(wav, hparams.quantize_channels - 1)
    else:
        out = wav

    # zero pad
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.pad_lr(out, hparams.fft_size, audio.get_hop_size())
    if l > 0 or r > 0:
        out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    # Write the spectrograms to disk:
    #name = splitext(basename(wav_path))[0]
    #audio_filename = '%s-wave.npy' % (name)
    #mel_filename = '%s-feats.npy' % (name)
    audio_filename = f'{out_dir}wave.npy'
    mel_filename = f'{out_dir}mel.npy'
    mfcc_filename = f'{out_dir}mfcc.npy'
    assert mfcc.shape[0] == N
    np.save(audio_filename,
            out.astype(out_dtype), allow_pickle=False)
    np.save(mel_filename,
            mel_spectrogram.astype(np.float32), allow_pickle=False)
    np.save(mfcc_filename,
            mfcc.astype(np.float32), allow_pickle=False)

    # Return a tuple describing this training example:
    return (out_dir, N, sp_ind,text)
if __name__ == "__main__":
    args = docopt(__doc__)
    #name = args["<name>"]
    scp_dir = args["<scp_dir>"] # default: scp/2019/[train,dev,test]_src_dst.json
    out_dir = args["<out_dir>"] # default: dump/2019/[english, surprise]/[train_no_dev,dev,test]/
    sp2ind_dir = args['<sp2ind_dir>'] # 2019_speaker2ind.json
    #num_workers = args["--num_workers"]
    #num_workers = cpu_count() // 2 if num_workers is None else int(num_workers)
    preset = args["--preset"]
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    print("Sampling frequency: {}".format(hparams.sample_rate))
    preprocess( scp_dir, out_dir, sp2ind_dir)

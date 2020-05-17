# coding: utf-8
"""
Inference Rep from trained Autoencoder.

usage: 
    inference_2017.py [options] seg <scp_dir> <feat> <frame_rate>  <checkpoint> <dst_dir> <lan>
    inference_2017.py [options] <scp_dir> <feat> <frame_rate>  <checkpoint> <dst_dir> <lan>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    -h, --help                        Show help message.
"""
from docopt import docopt
import json
import sys
import os
from os.path import dirname, join, basename, splitext
import torch
import numpy as np
from nnmnkwii import preprocessing as P
from tqdm import tqdm
import librosa

from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input

import audio
from hparams import hparams, hparams_debug_string

from autoencoders.autoencoder import Model as AE ,Model2 as AE2 ,Model4 as AE4
from autoencoders.cat_ae_model import Model as CatAE
from wavenet_vocoder import WaveNet
from autoencoders.wavenet_ae_model import AE as wvae
from autoencoders.wavenet_ae_model import VQVAE
#from train import build_model as build_wavenet_model
#from new_ae_train import build_model as build_autoencoder_model


torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def build_vqvae_model():
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    wavenet = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
        use_speaker_embedding=True,
    )
    model = VQVAE(wavenet=wavenet,c_in=39,hid=64, frame_rate = hparams.frame_rate, K = hparams.K, use_time_jitter = hparams.time_jitter)
    return model
def build_autoencoder_model():
    model = eval(hparams.name)(c_in = hparams.cin_channels,hid=64)
    return model
def build_wvae_model():
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = 64
    upsample_params["cin_pad"] = hparams.cin_pad
    wavenet = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=64,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
        use_speaker_embedding=True,
    )
    model = wvae(wavenet=wavenet,c_in=39,hid=64)
    return model
def build_wavenet_model():
    if is_mulaw_quantize(hparams.input_type):
        if hparams.out_channels != hparams.quantize_channels:
            raise RuntimeError(
                "out_channels must equal to quantize_chennels if input_type is 'mulaw-quantize'")
    if hparams.upsample_conditional_features and hparams.cin_channels < 0:
        s = "Upsample conv layers were specified while local conditioning disabled. "
        s += "Notice that upsample conv layers will never be used."
        warn(s)

    upsample_params = hparams.upsample_params
    upsample_params["cin_channels"] = hparams.cin_channels
    upsample_params["cin_pad"] = hparams.cin_pad
    model = WaveNet(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
        use_speaker_embedding=True,
    )
    return model


def _process_utterance(base_dir, f, frame_rate, model, dst_dir, seg):
    #feat_path = base_dir + feat + '.norm.npy'
    feat_path = base_dir + f + '.npy'
    if not os.path.exists(feat_path):
        raise Exception
    dirs = base_dir.split('/')
    assert len(dirs) == 7
    idx = dirs[-2]
    time = dirs[-3]
    lan = dirs[-4]
    subset = dirs[1]
    out_dir = dst_dir + f'{subset}/track1/{lan}/{time}/{idx}.txt'
    
    feat = np.load(feat_path)
    print(f"feat shape {feat.shape}",flush=True)
    n_feat = feat.shape[1]
    n_frames = feat.shape[0]
    feat = feat.T
    feat_tensor = torch.from_numpy(np.array([feat]))
    seg_len = hparams.max_time_steps // hparams.hop_size
    
    #seg = True

    if seg:
        res = []
        for start in range(0,feat_tensor.size(2) - seg_len +1 ,seg_len ):
            if start + 2 * seg_len >= feat_tensor.size(2):
                seg_feat = feat_tensor[:,:,start:]
            else:
                seg_feat = feat_tensor[:,:,start:start + seg_len]
            seg_feat = seg_feat.to(device)
            rep_tensor = model.encode(seg_feat)
            res.append(rep_tensor)
        res = torch.cat(res,2)[0]
        rep = res.data.cpu().numpy().transpose()
    else:
    
    
        feat_tensor = feat_tensor.to(device)
        rep_tensor = model.encode(feat_tensor)
        rep = rep_tensor.data.cpu().numpy()[0].transpose()
    #rec = cri(out,feat_tensor).item()
    #print(f"rec {rec}",flush=True)
    time_steps = []
    time_interval = 1 / int(frame_rate)
    win_len = hparams.win_length /hparams.sample_rate
    #for i in range(seg_rep_b * seg_rep_t):
    for i in range(rep.shape[0]):
        start = i * time_interval    
        end = start + win_len
        time_step = (start + end) / 2
        time_steps.append(time_step)
    time_steps_np = np.expand_dims(np.array(time_steps),axis=1)
    out = np.concatenate([time_steps_np,rep],axis=1)
    print(f"{out.shape}: {out_dir} ",flush=True)
    out_dir_dirname = os.path.dirname(out_dir)
    os.makedirs(out_dir_dirname,exist_ok=True)
    np.savetxt(out_dir, out,fmt='%.6f')

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]
    os.makedirs(dst_dir, exist_ok=True)
    scp_dir = args['<scp_dir>']
    feat = args['<feat>']
    frame_rate = args['<frame_rate>']
    lan = args['<lan>']
    seg = args['seg']
    #length = int(args["--length"])
    #initial_value = args["--initial-value"]
    #initial_value = None if initial_value is None else float(initial_value)
    #conditional_path = args["--conditional"]

    #file_name_suffix = args["--file-name-suffix"]
    #output_html = args["--output-html"]
    #speaker_id = args["--speaker-id"]
    #speaker_id = None if speaker_id is None else int(speaker_id)
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    print(hparams_debug_string())
    #assert hparams.name == "wavenet_vocoder"

    # Load conditional features
    #if conditional_path is not None:
    #    c = np.load(conditional_path)
    #    if c.shape[1] != hparams.num_mels:
    #        c = np.swapaxes(c, 0, 1)
    #else:
    #    c = None

    #from train import build_model

    # Model
    if hparams.name == 'wvae':
        model = build_wvae_model()
    elif hparams.name == 'vqvae':
        model = build_vqvae_model()
    else:
        model = build_autoencoder_model()#.to(device)
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    print("Load checkpoint from {}".format(checkpoint_path),flush=True)
    model.eval()
    scp_f = open(scp_dir)
    file_list = json.load(scp_f)
    
    for _,base_dir in file_list:
        if lan in base_dir:
            _process_utterance(base_dir,feat,frame_rate,model,dst_dir, seg)

# coding: utf-8
"""
Inference Rep from trained Autoencoder.

usage: inference_2019.py [options]  <scp_dir> <feat>  <checkpoint> <dst_dir>

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
from autoencoders.wavenet_ae_model import AE as wvae, INAE, INAE1
from autoencoders.wavenet_ae_model import VQVAE, CatWavAE
#from train import build_model as build_wavenet_model
#from new_ae_train import build_model as build_autoencoder_model


torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
def build_catae_model():
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
    model = CatWavAE(wavenet=wavenet,c_in=39,hid= hparams.cin_channels, tau = 0.1, k = hparams.K, frame_rate = hparams.frame_rate, hard = hparams.hard, slices = hparams.num_slices)
    return model
def build_inae_model():
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
    if hparams.name == 'inae':
        model = INAE(wavenet = wavenet, c_in = 39, hid = 64, frame_rate = hparams.frame_rate, adain = hparams.adain)
    elif hparams.name == 'inae1':

        model = INAE1(wavenet = wavenet, c_in = 39, hid = 64, frame_rate = hparams.frame_rate, adain = hparams.adain)
    return model
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
    if hparams.use_K1 and hparams.K1 != hparams.K:
        K1 = hparams.K1
    else:
        K1 = None
    if hparams.post_conv:
        hid = 64
    else:
        hid = hparams.cin_channels
    model = VQVAE(wavenet=wavenet,c_in=39,hid=hid, frame_rate = hparams.frame_rate, 
                use_time_jitter = hparams.time_jitter, K = hparams.K, ema = hparams.ema, 
                sliced = hparams.sliced, ins_norm = hparams.ins_norm, post_conv = hparams.post_conv, adain = hparams.adain,
                dropout = hparams.vq_drop, drop_dim = hparams.drop_dim, K1 = K1, num_slices = hparams.num_slices )
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
    model = wvae(wavenet=wavenet,c_in=39,hid=64, frame_rate = hparams.frame_rate)
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


def _process_utterance(base_dir, f, model, dst_dir):
    #feat_path = base_dir + feat + '.norm.npy'
    feat_path = base_dir + f + '.npy'
    if not os.path.exists(feat_path):
        raise Exception
    
    
    
    dirs = base_dir.split('/')
    assert len(dirs) == 6
    lan = dirs[-4]
    fnm = dirs[-2]
    out_dir = dst_dir + f'2019/{lan}/test/{fnm}.txt'
    
    feat = np.load(feat_path)
    
    print(f"feat shape {feat.shape}",flush=True)
    
    n_feat = feat.shape[1]
    n_frames = feat.shape[0]
    feat = feat.T
    feat_tensor = torch.from_numpy(np.array([feat]))
    seg_len = hparams.max_time_steps // hparams.hop_size
    


    feat_tensor = feat_tensor.to(device)
    rep_tensor = model.encode(feat_tensor)
    rep = rep_tensor.data.cpu().numpy()[0].transpose()
    out = rep
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
    elif hparams.name == 'inae' or hparams.name == 'inae1':
        model = build_inae_model()
    elif hparams.name == 'catae':
        model = build_catae_model()
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
        _process_utterance(base_dir,feat,model,dst_dir )

# coding: utf-8
"""
Synthesis waveform from trained WaveNet.

usage: synthesis.py [options] <dump_root> <checkpoint> <dst_dir> <syn_list> <speaker2ind> <lan> <up_factor> <frame_rate> <start_ind>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.
    --conditional=<p>                 Conditional features path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --speaker-id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
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
from hparams import hparams

from train import to_categorical
from wavenet_vocoder import WaveNet
from autoencoders.wavenet_ae_model import AE as wvae, INAE as inae, NewINAE as new_inae, INAE1 as inae1
from autoencoders.wavenet_ae_model import VQVAE

from autoencoders.wavenet_ae_model import CatWavAE

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
def build_new_inae_model():
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
        gin_channels=64,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        cin_pad=hparams.cin_pad,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_params=upsample_params,
        scalar_input=is_scalar_input(hparams.input_type),
        output_distribution=hparams.output_distribution,
        use_speaker_embedding=False,
    )
    model = new_inae(wavenet = wavenet, c_in = 39, hid = 64, frame_rate = hparams.frame_rate)
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
        model = inae(wavenet = wavenet, c_in = 39, hid = 64, frame_rate = hparams.frame_rate, adain =  hparams.adain)
    elif hparams.name == 'inae1':
        model = inae1(wavenet = wavenet, c_in = 39, hid = 64, frame_rate = hparams.frame_rate, adain =  hparams.adain)
    
    
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


    model = VQVAE(wavenet=wavenet,c_in=39, hid = hid, frame_rate = hparams.frame_rate, 
                use_time_jitter = hparams.time_jitter, K = hparams.K, ema = hparams.ema, 
                sliced = hparams.sliced, ins_norm = hparams.ins_norm, post_conv = hparams.post_conv, adain = hparams.adain,
                dropout = hparams.vq_drop, drop_dim = hparams.drop_dim, K1 = K1, num_slices = hparams.num_slices )
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

def batch_wavegen(model, c=None, g=None, fast=True, tqdm=tqdm):
    from train import sanity_check
    sanity_check(model, c, g)
    assert c is not None
    B = c.shape[0]
    model.eval()
    if fast:
        model.make_generation_fast_()

    # Transform data to GPU
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    if hparams.upsample_conditional_features:
        length = (c.shape[-1] - hparams.cin_pad * 2) * audio.get_hop_size()
    else:
        # already dupulicated
        length = c.shape[-1]

    with torch.no_grad():
        y_hat = model.incremental_forward(
            c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    if is_mulaw_quantize(hparams.input_type):
        # needs to be float since mulaw_inv returns in range of [-1, 1]
        y_hat = y_hat.max(1)[1].view(B, -1).float().cpu().data.numpy()
        for i in range(B):
            y_hat[i] = P.inv_mulaw_quantize(y_hat[i], hparams.quantize_channels - 1)
    elif is_mulaw(hparams.input_type):
        y_hat = y_hat.view(B, -1).cpu().data.numpy()
        for i in range(B):
            y_hat[i] = P.inv_mulaw(y_hat[i], hparams.quantize_channels - 1)
    else:
        y_hat = y_hat.view(B, -1).cpu().data.numpy()

    if hparams.postprocess is not None and hparams.postprocess not in ["", "none"]:
        for i in range(B):
            y_hat[i] = getattr(audio, hparams.postprocess)(y_hat[i])

    if hparams.global_gain_scale > 0:
        for i in range(B):
            y_hat[i] /= hparams.global_gain_scale

    return y_hat


def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm, up_factor = 320, tar_c = None):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    #from train import sanity_check
    #sanity_check(model.wavenet, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)
    if tar_c is not None:
        tar_c = _to_numpy(tar_c)
    
    model.eval()
    if fast:
        model.wavenet.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        #upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * int(up_factor)
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not hparams.upsample_conditional_features:
            c = np.repeat(c, up_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)
    if tar_c is not None:
        tar_c = torch.FloatTensor(tar_c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(hparams.input_type):
            initial_value = P.mulaw_quantize(0, hparams.quantize_channels - 1)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(hparams.input_type):
        assert initial_value >= 0 and initial_value < hparams.quantize_channels
        #initial_input = np_utils.to_categorical(
        #    initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = to_categorical(
            initial_value, num_classes=hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)
    tar_c = None if tar_c is None else tar_c.to(device)
        
    
    with torch.no_grad():
        if tar_c is None:
            y_hat = model.incremental_forward(
                initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=hparams.log_scale_min)
        else:
            y_hat = model.incremental_forward(
                initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
                log_scale_min=hparams.log_scale_min, tar_c = tar_c)

    if is_mulaw_quantize(hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, hparams.quantize_channels)
    elif is_mulaw(hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    if hparams.postprocess is not None and hparams.postprocess not in ["", "none"]:
        y_hat = getattr(audio, hparams.postprocess)(y_hat)

    if hparams.global_gain_scale > 0:
        y_hat /= hparams.global_gain_scale

    return y_hat


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    dump_root = args['<dump_root>']
    checkpoint_path = args["<checkpoint>"]
    dst_dir = args["<dst_dir>"]
    syn_list_path = args["<syn_list>"]
    lan = args['<lan>']
    up_factor = args['<up_factor>']
    speaker2ind_path = args['<speaker2ind>']
    frame_rate = args['<frame_rate>']
    
    start_ind = args['<start_ind>']
    
    syn_l_f = open(syn_list_path, 'r')
    syn_list = syn_l_f.readlines()
    print(f"get {len(syn_list)} to syn", flush=True)

    sp_f = open(speaker2ind_path, 'r')
    sp2ind = json.load(sp_f)

    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)
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
    #model = build_model().to(device)

    if hparams.name == 'wvae':
        model = build_wvae_model().to(device)
    elif hparams.name == 'vqvae':
        model = build_vqvae_model().to(device)
    elif hparams.name == 'inae' or hparams.name == 'inae1':
        model = build_inae_model().to(device)
    elif hparams.name == 'new_inae':
        model = build_new_inae_model().to(device)
    elif hparams.name == 'catae':
        model = build_catae_model().to(device)
    else:
        raise Exception(f"no model {hparams.name}")
    # Load checkpoint
    print("Load checkpoint from {}".format(checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]
    print(model)
    for ind_syn  in range(int(start_ind), len(syn_list)):
    #for ind_syn  in [0, 22, 147]:
        
        line = syn_list[ind_syn]
        p, tar = line.strip().split()
        
        if lan == 'surprise':
            p = 'test/' + p
        
        fid = p.split('_')[1]
        if os.path.exists(f'{dump_root}/{p}/mfcc.norm.npy'):
            c = np.load(f'{dump_root}/{p}/mfcc.norm.npy')
            print(f"load feat {c.shape}",flush=True)
            div = 100 // int(frame_rate)
            if c.shape[0] % div != 0:
                pad_len = div - (c.shape[0] % div)
                c = np.pad(c, [[0, pad_len],[0,0]], mode='constant', constant_values = 0.)
                print(f"pad c into {c.shape}",flush=True)
        else:
            raise Exception(f'cant find con file in {dump_root}/{p}/mfcc.norm.npy ')      
        print(f"processing sou {p} to tar {tar}",flush=True)
        
        if tar in sp2ind:
            spid = sp2ind[tar]
        else:
            raise Exception(f'cant find sp {tar} in sp2ind {speaker2ind_path}')
        
        tar_c = None
        if hparams.name == 'inae' or hparams.name == 'new_inae' or hparams.name == 'inae1':
            if lan == 'english':
                if tar == 'V002':
                    tar_c_dir = 'dump/2019/english/train_no_dev/V002_4290703572/mfcc.norm.npy'
                    tar_c = np.load(tar_c_dir)

                elif tar == 'V001':
                    tar_c_dir = 'dump/2019/english/train_no_dev/V001_2817465453/mfcc.norm.npy'
                    tar_c = np.load(tar_c_dir)
            elif lan == 'surprise' :
                if tar == 'V001' :
                    tar_c_dir = 'dump/2019/surprise/train_no_dev/V001_115929/mfcc.norm.npy'
                    tar_c = np.load(tar_c_dir)
                else:
                    raise ValueError(f"tar {tar} not found")
            else:
                raise Exception
        
            print(f'using target speech from {tar_c_dir}', flush=True)
        #dst_wav_path = join(dst_dir, "{}{}.wav".format(checkpoint_name, file_name_suffix))
        dst_wav_path = f"{dst_dir}2019/{lan}/test/{tar}_{fid}.wav"
        os.makedirs(f'{dst_dir}2019/{lan}/test/', exist_ok=True)
        # DO generate
        waveform = wavegen(model, length, c=c, g=spid, initial_value=initial_value, fast=True, up_factor = up_factor, tar_c = tar_c)

        # save
        librosa.output.write_wav(dst_wav_path, waveform, sr=hparams.sample_rate)

        print("ind {} Finished! Check out {} for generated audio samples.".format(ind_syn, dst_wav_path))
    sys.exit(0)

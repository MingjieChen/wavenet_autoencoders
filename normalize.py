# coding: utf-8
"""Perform meanvar normalization to preprocessed features.

usage: preprocess_normalize.py [options] <scp_dir> <feat> <scaler>

options:
    --inverse                Inverse transform.
    -h, --help               Show help message.
"""
from docopt import docopt
import os
from os.path import join, exists, basename, splitext
from tqdm import tqdm
from nnmnkwii import preprocessing as P
import numpy as np
import json
from functools import partial
from shutil import copyfile
import json
import joblib
from glob import glob
from itertools import zip_longest
def get_paths_by_glob(in_dir, filt):
    return glob(join(in_dir, filt))


def _process_utterance(dst_path,feat, scaler, inverse):
    # [Optional] copy audio with the same name if exists
    #if audio_path is not None and exists(audio_path):
    #    name = splitext(basename(audio_path))[0]
    #    np.save(join(out_dir, name), np.load(audio_path), allow_pickle=False)
    feat_path = dst_path + feat +'.npy'
    norm_path = dst_path + feat + '.norm'
    # [Required] apply normalization for features
    if not inverse:
        assert exists(feat_path)
    else:
        assert exists(norm_path)
    if not inverse:
        x = np.load(feat_path)
    else:
        x = np.load(norm_path)
    if inverse:
        y = scaler.inverse_transform(x)
    else:
        y = scaler.transform(x)
    assert x.dtype == y.dtype
    #name = splitext(basename(feat_path))[0]
    if not inverse:
        np.save(norm_path, y, allow_pickle=False)
    else:
        np.save(feat_path, y, allow_pickle=False)


def apply_normalization_dir2dir(scp_dir, feat, scaler, inverse):
    # NOTE: at this point, audio_paths can be empty
    #audio_paths = get_paths_by_glob(in_dir, "*-wave.npy")
    #feature_paths = get_paths_by_glob(in_dir, "*-feats.npy")
    #executor = ProcessPoolExecutor(max_workers=num_workers)
    #futures = []
    #for audio_path, feature_path in zip_longest(audio_paths, feature_paths):
        #futures.append(executor.submit(
        #    partial(_process_utterance, out_dir, audio_path, feature_path, scaler, inverse)))
    #    _process_utterance(out_dir,audio_path,feature_path,scaler,inverse)
    #for future in tqdm(futures):
    #    future.result()

    f = open(scp_dir)
    file_list = json.load(f)
    for src,dst in file_list:
        _process_utterance(dst,feat,scaler,inverse)


if __name__ == "__main__":
    args = docopt(__doc__)
    scp_dir = args["<scp_dir>"]
    feat = args['<feat>']
    #out_dir = args["<out_dir>"]
    scaler_path = args["<scaler>"]
    scaler = joblib.load(scaler_path)
    inverse = args["--inverse"]
    apply_normalization_dir2dir(scp_dir, feat, scaler, inverse)


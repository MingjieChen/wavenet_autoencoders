"""Trainining script for AutoEncoder

usage: ae_train.py [options]

options:
    --dump-root=<dir>            Directory contains preprocessed features.
    --checkpoint-dir=<dir>       Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --restore-parts=<path>       Restore part of the model.
    --log-event-path=<name>      Log event path.
    --reset-optimizer            Reset optimizer.
    --use-norm                   Use Normalized feat.
    --feat=<name>                Feature type.
    --speaker-id=<N>             Use specific speaker of data in case for multi-speaker datasets.
    -h, --help                   Show this help message and exit
"""
from docopt import docopt

import sys

import os
from os.path import dirname, join, expanduser, exists
from tqdm import tqdm
from datetime import datetime
import random
import json
from glob import glob

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import lrschedule

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.utils.data.sampler import Sampler

from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import librosa.display

from tensorboardX import SummaryWriter
from matplotlib import cm
from warnings import warn

#from wavenet_vocoder import WaveNet
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw, is_scalar_input
#from wavenet_vocoder.mixture import discretized_mix_logistic_loss
#from wavenet_vocoder.mixture import sample_from_discretized_mix_logistic
#from wavenet_vocoder.mixture import mix_gaussian_loss
#from wavenet_vocoder.mixture import sample_from_mix_gaussian

import audio
from hparams import hparams, hparams_debug_string
from autoencoders.autoencoder import Model as AE ,Model2 as AE2 ,Model4 as AE4
from autoencoders.cat_ae_model import Model as CatAE
global_step = 0
global_test_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = True
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint
def _pad(seq, max_len, constant_values=0):
    return np.pad(seq, (0, max_len - len(seq)),
                  mode='constant', constant_values=constant_values)


def _pad_2d(x, max_len, b_pad=0, constant_values=0):
    x = np.pad(x, [(b_pad, max_len - len(x) - b_pad), (0, 0)],
               mode="constant", constant_values=constant_values)
    return x
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
# TODO: I know this is too ugly...
class _NPYDataSource(FileDataSource):
    def __init__(self, dump_root, norm=False, typ="mel", speaker_id=None, max_steps=8000,
                 cin_pad=0, hop_size=160):
        self.dump_root = dump_root
        #self.col = col
        self.lengths = []
        self.speaker_id = speaker_id
        self.multi_speaker = False
        self.speaker_ids = None
        self.max_steps = max_steps
        self.cin_pad = cin_pad
        self.hop_size = hop_size
        self.typ = typ
        self.norm=norm

    def collect_files(self):
        meta = join(self.dump_root, "train.txt")
        if not exists(meta):
            #paths = sorted(glob(join(self.dump_root, "*-{}.npy".format(self.typ))))
            #return paths
            raise Exception(f'{meta} does not exist')

        with open(meta, "rb") as f:
            lines = f.readlines()
        l = lines[0].decode("utf-8").split("|")
        #assert len(l) == 4 or len(l) == 5
        self.multi_speaker = int(l[2])!=-1
        self.lengths = list(
            map(lambda l: int(l.decode("utf-8").split("|")[1]), lines))
        
        if self.typ == 'mel' or self.typ == 'mfcc':
            if self.norm:
                paths = list(map(lambda l: l.decode("utf-8").split("|")[0]+f'{self.typ}.norm.npy', lines))
            else:
                paths = list(map(lambda l: l.decode("utf-8").split("|")[0] +f'{self.typ}.npy' , lines ))
        else:
            paths = list(map(lambda l: l.decode("utf-8").split("|")[0] +f'{self.typ}.npy' , lines ))



        #paths = list(map(lambda f: join(self.dump_root, f), paths_relative))

        # Exclude small files (assuming lenghts are in frame unit)
        # TODO: consider this for multi-speaker
        if self.max_steps is not None:
            idx = np.array(self.lengths) * self.hop_size > self.max_steps + 2 * self.cin_pad * self.hop_size
            if idx.sum() != len(self.lengths):
                print("{} short samples are omitted for training.".format(len(self.lengths) - idx.sum()))
            self.lengths = list(np.array(self.lengths)[idx])
            paths = list(np.array(paths)[idx])

        if self.multi_speaker:
            speaker_ids = list(map(lambda l: int(l.decode("utf-8").split("|")[-2]), lines))
            self.speaker_ids = speaker_ids
            if self.speaker_id is not None:
                # Filter by speaker_id
                # using multi-speaker dataset as a single speaker dataset
                indices = np.array(speaker_ids) == self.speaker_id
                paths = list(np.array(paths)[indices])
                self.lengths = list(np.array(self.lengths)[indices])
                # aha, need to cast numpy.int64 to int
                self.lengths = list(map(int, self.lengths))
                self.multi_speaker = False

#!!!changed from source code
        #if self.multi_speaker:
                speaker_ids_np = list(np.array(self.speaker_ids)[indices])
                self.speaker_ids = list(map(int, speaker_ids_np))
                assert len(paths) == len(self.speaker_ids)

        return paths

    def collect_features(self, path):
        return np.load(path)
class PartialyRandomizedSimilarTimeLengthSampler(Sampler):
    """Partially randomized sampler

    1. Sort by lengths
    2. Pick a small patch and randomize it
    3. Permutate mini-batches
    """

    def __init__(self, lengths, batch_size=8, batch_group_size=None):
        self.lengths, self.sorted_indices = torch.sort(torch.LongTensor(lengths))

        self.batch_size = batch_size
        if batch_group_size is None:
            batch_group_size = min(batch_size * 8, len(self.lengths))
            if batch_group_size % batch_size != 0:
                batch_group_size -= batch_group_size % batch_size

        self.batch_group_size = batch_group_size
        assert batch_group_size % batch_size == 0

    def __iter__(self):
        indices = self.sorted_indices.numpy()
        batch_group_size = self.batch_group_size
        s, e = 0, 0
        bins = []
        for i in range(len(indices) // batch_group_size):
            s = i * batch_group_size
            e = s + batch_group_size
            group = indices[s:e]
            random.shuffle(group)
            bins += [group]

        # Permutate batches
        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        # Handle last elements
        s += batch_group_size
        if s < len(indices):
            last_bin = indices[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.sorted_indices)
class RawAudioDataSource(_NPYDataSource):
    def __init__(self, dump_root, **kwargs):
        super(RawAudioDataSource, self).__init__(dump_root, False, "wave", **kwargs)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self, dump_root, norm=True, typ = 'mel',**kwargs):
        super(MelSpecDataSource, self).__init__(dump_root, norm, typ, **kwargs)
def maybe_set_epochs_based_on_max_steps(hp, steps_per_epoch):
    nepochs = hp.nepochs
    max_train_steps = hp.max_train_steps
    if max_train_steps is not None:
        epochs = int(np.ceil(max_train_steps / steps_per_epoch))
        hp.nepochs = epochs
        print("info; Number of epochs is set based on max_train_steps: {}".format(epochs))
# https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3
def restore_parts(path, model):
    print("Restore part of the model from: {}".format(path))
    state = _load(path)["state_dict"]
    model_dict = model.state_dict()
    valid_state_dict = {k: v for k, v in state.items() if k in model_dict}

    try:
        model_dict.update(valid_state_dict)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        # there should be invalid size of weight(s), so load them per parameter
        print(str(e))
        model_dict = model.state_dict()
        for k, v in valid_state_dict.items():
            model_dict[k] = v
            try:
                model.load_state_dict(model_dict)
            except RuntimeError as e:
                print(str(e))
                warn("{}: may contain invalid size of weight. skipping...".format(k))
def assert_ready_for_upsampling(x, c, cin_pad):
    assert len(x) == (len(c) - 2 * cin_pad) * audio.get_hop_size()
class PyTorchDataset(object):
    def __init__(self, X, Mel):
        self.X = X
        self.Mel = Mel
        # alias
        self.multi_speaker = X.file_data_source.multi_speaker

    def __getitem__(self, idx):
        if self.Mel is None:
            mel = None
        else:
            mel = self.Mel[idx]

        raw_audio = self.X[idx]
        if self.multi_speaker:
            speaker_id = self.X.file_data_source.speaker_ids[idx]
        else:
            speaker_id = None

        # (x,c,g)
        return raw_audio, mel, speaker_id

    def __len__(self):
        return len(self.X)
def collate_fn(batch):
    """Create batch

    Args:
        batch(tuple): List of tuples
            - x[0] (ndarray,int) : list of (T,)
            - x[1] (ndarray,int) : list of (T, D)
            - x[2] (ndarray,int) : list of (1,), speaker id
    Returns:
        tuple: Tuple of batch
            - x (FloatTensor) : Network inputs (B, C, T)
            - y (LongTensor)  : Network targets (B, T, 1)
    """

    local_conditioning = len(batch[0]) >= 2 and hparams.cin_channels > 0
    global_conditioning = len(batch[0]) >= 3 and hparams.gin_channels > 0

    if hparams.max_time_sec is not None:
        max_time_steps = int(hparams.max_time_sec * hparams.sample_rate)
    elif hparams.max_time_steps is not None:
        max_time_steps = hparams.max_time_steps
    else:
        max_time_steps = None

    # Time resolution adjustment
    cin_pad = hparams.cin_pad
    if local_conditioning:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            if hparams.upsample_conditional_features:
                assert_ready_for_upsampling(x, c, cin_pad=0)
                if max_time_steps is not None:
                    max_steps = ensure_divisible(max_time_steps, audio.get_hop_size(), True)
                    if len(x) > max_steps:
                        max_time_frames = max_steps // audio.get_hop_size()
                        s = np.random.randint(cin_pad, len(c) - max_time_frames - cin_pad)
                        ts = s * audio.get_hop_size()
                        x = x[ts:ts + audio.get_hop_size() * max_time_frames]
                        c = c[s - cin_pad:s + max_time_frames + cin_pad, :]
                        assert_ready_for_upsampling(x, c, cin_pad=cin_pad)
            else:
                x, c = audio.adjust_time_resolution(x, c)
                if max_time_steps is not None and len(x) > max_time_steps:
                    s = np.random.randint(cin_pad, len(x) - max_time_steps - cin_pad)
                    x = x[s:s + max_time_steps]
                    c = c[s - cin_pad:s + max_time_steps + cin_pad, :]
                assert len(x) == len(c)
            new_batch.append((x, c, g))
        batch = new_batch
    else:
        new_batch = []
        for idx in range(len(batch)):
            x, c, g = batch[idx]
            x = audio.trim(x)
            if max_time_steps is not None and len(x) > max_time_steps:
                s = np.random.randint(0, len(x) - max_time_steps)
                if local_conditioning:
                    x, c = x[s:s + max_time_steps], c[s:s + max_time_steps, :]
                else:
                    x = x[s:s + max_time_steps]
            new_batch.append((x, c, g))
        batch = new_batch

    # Lengths
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = max(input_lengths)

    # (B, T, C)
    # pad for time-axis
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        x_batch = np.array([_pad_2d(to_categorical(
            x[0], num_classes=hparams.quantize_channels),
            max_input_len, 0, padding_value) for x in batch], dtype=np.float32)
    else:
        x_batch = np.array([_pad_2d(x[0].reshape(-1, 1), max_input_len)
                            for x in batch], dtype=np.float32)
    assert len(x_batch.shape) == 3

    # (B, T)
    if is_mulaw_quantize(hparams.input_type):
        padding_value = P.mulaw_quantize(0, mu=hparams.quantize_channels - 1)
        y_batch = np.array([_pad(x[0], max_input_len, constant_values=padding_value)
                            for x in batch], dtype=np.int)
    else:
        y_batch = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.float32)
    assert len(y_batch.shape) == 2

    # (B, T, D)
    if local_conditioning:
        max_len = max([len(x[1]) for x in batch])
        c_batch = np.array([_pad_2d(x[1], max_len) for x in batch], dtype=np.float32)
        assert len(c_batch.shape) == 3
        # (B x C x T)
        c_batch = torch.FloatTensor(c_batch).transpose(1, 2).contiguous()
    else:
        c_batch = None

    if global_conditioning:
        g_batch = torch.LongTensor([x[2] for x in batch])
    else:
        g_batch = None

    # Covnert to channel first i.e., (B, C, T)
    x_batch = torch.FloatTensor(x_batch).transpose(1, 2).contiguous()
    # Add extra axis
    if is_mulaw_quantize(hparams.input_type):
        y_batch = torch.LongTensor(y_batch).unsqueeze(-1).contiguous()
    else:
        y_batch = torch.FloatTensor(y_batch).unsqueeze(-1).contiguous()

    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, c_batch, g_batch, input_lengths
def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch, ema=None):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    global global_test_step
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

    import shutil
    latest_pth = join(checkpoint_dir, "checkpoint_latest.pth")
    shutil.copyfile(checkpoint_path, latest_pth)

    if ema is not None:
        averaged_model = clone_as_averaged_model(device, model, ema)
        checkpoint_path = join(
            checkpoint_dir, "checkpoint_step{:09d}_ema.pth".format(global_step))
        torch.save({
            "state_dict": averaged_model.state_dict(),
            "optimizer": optimizer_state,
            "global_step": step,
            "global_epoch": epoch,
            "global_test_step": global_test_step,
        }, checkpoint_path)
        print("Saved averaged checkpoint:", checkpoint_path)

        latest_pth = join(checkpoint_dir, "checkpoint_latest_ema.pth")
        shutil.copyfile(checkpoint_path, latest_pth)
def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    #global_test_step = checkpoint.get("global_test_step", 0)
    global_test_step = checkpoint["global_test_step"]

    return model
def ensure_divisible(length, divisible_by=256, lower=True):
    if length % divisible_by == 0:
        return length
    if lower:
        return length - length % divisible_by
    else:
        return length + (divisible_by - length % divisible_by)
def get_data_loaders(dump_root, speaker_id, test_shuffle=True,norm=True,typ='mel'):
    data_loaders = {}
    local_conditioning = hparams.cin_channels > 0

    if hparams.max_time_steps is not None:
        max_steps = ensure_divisible(hparams.max_time_steps, audio.get_hop_size(), True)
    else:
        max_steps = None

    for phase in ["train_no_dev", "dev"]:
        train = phase == "train_no_dev"
        X = FileSourceDataset(
            RawAudioDataSource(join(dump_root, phase), speaker_id=speaker_id,
                               max_steps=max_steps, cin_pad=hparams.cin_pad,
                               hop_size=audio.get_hop_size()))
        if local_conditioning:
            Mel = FileSourceDataset(
                MelSpecDataSource(join(dump_root, phase), norm=norm,typ = typ,speaker_id=speaker_id,
                                  max_steps=max_steps, cin_pad=hparams.cin_pad,
                                  hop_size=audio.get_hop_size()))
            #assert len(X) == len(Mel)
            print("Local conditioning enabled. Shape of a sample: {}.".format(
                Mel[0].shape))
        else:
            Mel = None
        print("[{}]: length of the dataset is {}".format(phase, len(X)))

        if train:
            lengths = np.array(X.file_data_source.lengths)
            # Prepare sampler
            sampler = PartialyRandomizedSimilarTimeLengthSampler(
                lengths, batch_size=hparams.batch_size)
            shuffle = False
            # make sure that there's no sorting bugs for https://github.com/r9y9/wavenet_vocoder/issues/130
            sampler_idx = np.asarray(sorted(list(map(lambda s: int(s), sampler))))
            assert (sampler_idx == np.arange(len(sampler_idx), dtype=np.int)).all()
        else:
            sampler = None
            shuffle = test_shuffle

        dataset = PyTorchDataset(X, Mel)
        if train:
            collate_function = collate_fn
            batch_size = hparams.batch_size
            data_loader = data_utils.DataLoader(
                dataset, batch_size=batch_size, drop_last=False,
                num_workers=hparams.num_workers, sampler=sampler, shuffle=shuffle,
                collate_fn=collate_function, pin_memory=hparams.pin_memory)
        else:
            batch_size=1
            data_loader = data_utils.DataLoader(
                dataset, batch_size=batch_size, drop_last=False,
                sampler=sampler, shuffle=shuffle,
                pin_memory=hparams.pin_memory)
        

        speaker_ids = {}
        if X.file_data_source.multi_speaker:
            for idx, (x, c, g) in enumerate(dataset):
                if g is not None:
                    try:
                        speaker_ids[g] += 1
                    except KeyError:
                        speaker_ids[g] = 1
            if len(speaker_ids) > 0:
                print("Speaker stats:", speaker_ids)

        data_loaders[phase] = data_loader

    return data_loaders

def build_model():
    model = eval(hparams.name)(c_in = hparams.cin_channels,hid=64)
    return model
    
def train_fn(device,model,data_loaders,optimizer,writer,checkpoint_dir=None):
    criterion = nn.MSELoss()
    
    global global_step,global_epoch,global_test_step

    while global_epoch < hparams.nepochs:
        #for phase,data_loader in data_loaders.items():
        #is_training = (phase=='train_no_dev')
        running_loss = 0.
        for step, (x,y,c,g,input_lengths) in enumerate(data_loaders['train_no_dev']):
            #do_eval = False
            #if is_training and global_epoch % hparams.test_eval_epoch_interval == 0:
            #    do_eval = True
            #if do_eval:
            #    print(f"{phase} Eval at train step {global_step}")

            running_loss += __train_step(device, global_epoch, global_step, global_test_step, model, optimizer,writer,criterion,x,y,c,g,input_lengths, checkpoint_dir)

            #if is_training:
            global_step +=1
            #else:
            #    global_test_step += 1

            if global_step >= hparams.max_train_steps:
                print(f"training reach max train steps {hparams.max_train_steps}, will exit")
                return
        
        averaged_loss = running_loss / len(data_loaders['train_no_dev'])
        writer.add_scalar(f"train_no_dev loss (per epoch)", averaged_loss,global_epoch)
        print(f"Epoch {global_epoch} [train_no_dev] Loss: {averaged_loss}")
        
        
        #do eval for the whole dataloader
        if global_epoch % hparams.test_eval_epoch_interval == 0:
            global_test_step +=1
            eval_model(global_epoch, global_test_step, writer,device,model,criterion,data_loaders['dev'])
        
        
        
        global_epoch += 1
    return

def __train_step(device, epoch, global_step, global_test_step, model, optimizer, writer, criterion, x, y, c, g, input_lengths, checkpoint_dir  ):
    #is_training = (phase == 'train_no_dev')
    clip_thresh = hparams.clip_thresh

    #if is_training:
    model.train()
    step = global_step
    #else:
    #    model.eval()
    #    step = global_test_step
    current_lr = hparams.optimizer_params['lr']
    if hparams.lr_schedule is not None:
        lr_schedule_f = getattr(lrschedule, hparams.lr_schedule)
        current_lr = lr_schedule_f(hparams.optimizer_params['lr'], step, **hparams.lr_schedule_kwargs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    optimizer.zero_grad()

    c = c.to(device)
    c_hat = model(c)
    
    loss = criterion(c_hat, c)

    if step >0 and step % hparams.checkpoint_interval == 0:
        save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch)


    #if do_eval:
    #    eval_model(global_step, writer, device, model, y, c, g, input_lengths)
    #if is_training:
    loss.backward()
    if clip_thresh >0 :
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_thresh)
    optimizer.step()
    writer.add_scalar(f"train step loss", loss.item(), step)
    return loss.item()


def eval_model(global_epoch,global_test_step, writer, device, model,criterion ,data_loader):
    model.eval()
    eval_loss = 0.


    #make c into batch
    max_frames = hparams.max_time_steps // audio.get_hop_size()
    for x, c, g  in data_loader:
        base_batches = c.size(1) // max_frames
        cut_length = max_frames * base_batches
        #print(f"eval cut_length {cut_length} base_batches {base_batches}")
        start_idxs = np.arange(0,base_batches,64)
        cut_c = c[:,:cut_length,:]
        whole_batch_c = cut_c.view(base_batches,hparams.cin_channels,max_frames)

        sample_eval_loss = 0.
        for start_idx in start_idxs:
            batch_c = whole_batch_c[start_idx:start_idx+64,:,:]
            batch_c = batch_c.to(device)
            batch_c_hat = model(batch_c)
            sample_eval_loss += criterion(batch_c_hat,batch_c).item()
        avg_sample_eval_loss = sample_eval_loss / len(start_idxs)
        eval_loss += avg_sample_eval_loss
    averaged_eval_loss = eval_loss / len(data_loader)
    writer.add_scalar(f'Epoch eval loss',averaged_eval_loss,global_test_step)
    print(f"Epoch {global_epoch} [dev] Loss: {averaged_eval_loss}")
    
    

    

if __name__ == "__main__":

    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    checkpoint_restore_parts = args["--restore-parts"]
    #speaker_id = args["--speaker-id"]
    speaker_id = None
    preset = args["--preset"]

    dump_root = args["--dump-root"]
    norm = args["--use-norm"]
    typ = args["--feat"]
    
    
    #if dump_root is None:
        #dump_root = join(dirname(__file__), "data", "ljspeech")

    log_event_path = args["--log-event-path"]
    reset_optimizer = args["--reset-optimizer"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    #assert hparams.name == "wavenet_vocoder"
    print(hparams_debug_string())

    fs = hparams.sample_rate

    os.makedirs(checkpoint_dir, exist_ok=True)

    output_json_path = join(checkpoint_dir, "hparams.json")
    with open(output_json_path, "w") as f:
        json.dump(hparams.values(), f, indent=2)

    # Dataloader setup
    data_loaders = get_data_loaders(dump_root, norm=norm,typ=typ,speaker_id=speaker_id, test_shuffle=False)
    maybe_set_epochs_based_on_max_steps(hparams, len(data_loaders["train_no_dev"]))
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model().to(device)
    from torch import optim
    Optimizer = getattr(optim, hparams.optimizer)
    optimizer = Optimizer(model.parameters(), **hparams.optimizer_params)

    if checkpoint_restore_parts is not None:
        restore_parts(checkpoint_restore_parts, model)

    # Load checkpoints
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

    # Setup summary writer for tensorboard
    if log_event_path is None:
        log_event_path = "log/run-test" + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path: {}".format(log_event_path))
    writer = SummaryWriter(log_dir=log_event_path)
    train_fn(device,model,data_loaders,optimizer,writer,checkpoint_dir)

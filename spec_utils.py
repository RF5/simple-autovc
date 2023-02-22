import os
import pickle
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
from numpy.random import RandomState
import torch
import librosa
import argparse
from hp import hp
import math

def pad_seq(x, base=32):
    len_out = int(base * math.ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return torch.nn.functional.pad(x, (0,0,0,len_pad), value=0), len_pad


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def pySTFT(x, fft_length=1024, hop_length=256):
    x = np.pad(x, int(fft_length // 2), mode='reflect')
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape,strides=strides)
    fft_window = get_window('hann', fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)

mel_basis = mel(hp.sampling_rate, 1024, fmin=90, fmax=7600, n_mels=80).T # For wavenet vocoder
mel_basis_hifi = mel(hp.sampling_rate, 1024, fmin=0, fmax=8000, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, hp.sampling_rate, order=5)

def get_mspec(fn, is_hifigan=True, return_waveform=False):
    # Read audio file
    x, fs = sf.read(str(fn))
    x = x[:, 0]
    print(x.dtype, x.shape, fs)

    x = librosa.resample(x, fs, hp.sampling_rate)
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    print(y.shape)
    wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0])-0.5)*1e-06
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    #print(D.shape)
    if is_hifigan:
        D_mel = np.dot(D, mel_basis_hifi)
        S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)
    else:
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1) # y = (x + 100) / 100
        S = S.astype(np.float32)

    if return_waveform: return torch.from_numpy(S), y
    return torch.from_numpy(S)

def get_mspec_from_array(x, input_sr, is_hifigan=True, return_waveform=False):
    """ `x` must be a 1D numpy array corresponding to a waveform"""
    #print(x.dtype, x.shape, fs)
    x = librosa.resample(x, input_sr, hp.sampling_rate)
    # Remove drifting noise
    y = signal.filtfilt(b, a, x)
    # Ddd a little random noise for model roubstness
    wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0])-0.5)*1e-06
    # Compute spect
    D = pySTFT(wav).T
    # Convert to mel and normalize
    #print(D.shape)
    if is_hifigan:
        D_mel = np.dot(D, mel_basis_hifi)
        S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)
    else:
        D_mel = np.dot(D, mel_basis)
        D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
        S = np.clip((D_db + 100) / 100, 0, 1) # y = (x + 100) / 100
        S = S.astype(np.float32)

    if return_waveform: return torch.from_numpy(S), y
    return torch.from_numpy(S)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autovc trainer')
    parser.add_argument('--out_mel_dir', action='store', required=True,
                        help='output directory to save mel spectrograms')

    args = parser.parse_args()
    out_path = Path(args.out_mel_dir)
    os.makedirs(out_path, exist_ok=True)

    from fastprogress import progress_bar, master_bar
    from fastcore.parallel import parallel
    import random

    print("Precomputing spectrograms.")
    ds_root = Path(hp.data_root)
    spk_folders = sorted(list((ds_root/'wav48_silence_trimmed').iterdir()))
    print(f"[DATA] Found a total of {len(spk_folders)} speakers")

    random.seed(hp.seed)
    train_spk_folders = sorted(random.sample(spk_folders, k=hp.n_train_speakers))
    test_spk_folders = sorted(list(set(spk_folders) - set(train_spk_folders)))
    train_files = []
    for pth in train_spk_folders: train_files.extend(list(pth.iterdir()))
    test_files = []
    for pth in test_spk_folders: test_files.extend(list(pth.iterdir()))
    print(f"[DATA] Split into {len(train_spk_folders)} train speakers ({len(train_files)} files)")
    print(f"[DATA] and {len(test_spk_folders)} test speakers ({len(test_files)} files)")

    def proc_spec(path):
        mspec = get_mspec(path, is_hifigan=True)
        os.makedirs(out_path/path.parent.name, exist_ok=True)
        torch.save(mspec, out_path/path.parent.name/f"{path.stem}.pt")

    print("Precomputing train spectrograms")
    parallel(proc_spec, train_files, progress=True)
    print("Precomputing test spectrograms")
    parallel(proc_spec, test_files, progress=True)
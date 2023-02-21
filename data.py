from torch.utils import data
import torch.nn.functional as F
import torch
import numpy as np
import pickle 
import os
from spec_utils import get_mspec
import random
       
class AutoVCDataset(data.Dataset):

    def __init__(self, paths, spk_embs, len_crop, scale=None, shift=None) -> None:
        super().__init__()
        self.paths = paths
        self.spk_embs = spk_embs
        self.len_crop = len_crop
        # assert jitter % 32 == 0, "Jitter must be divisible by 32"
        # self.jitter_choices = list(range(0, jitter+1, 32))

        # Replacements for lambda function to fix AttributeError
        global norm_mel
        def norm_mel(x):
            return (x + shift) / scale
        global denorm_mel
        def denorm_mel(x):
            return (x * scale) - shift
        global identity
        def identity(x):
            return x

        if scale is not None and shift is not None:
            self.norm_mel = norm_mel
            self.denorm_mel = denorm_mel
        else:
            self.norm_mel = identity
            self.denorm_mel = identity
        
        

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index):
        pth = self.paths[index]
        if pth.suffix == '.pt': mspec = torch.load(str(pth)) # (N, n_mels)
        else: mspec = get_mspec(pth, is_hifigan=True) # (N, n_mels)
        mspec = self.random_crop(mspec)
        spk_id = pth.parent.stem
        spk_emb = self.spk_embs[spk_id]
        mspec = self.norm_mel(mspec)
        return mspec, spk_emb

    def random_crop(self, mspec):
        #cprint(mspec.shape) 
        N, _ = mspec.shape
        clen = self.len_crop
        if N < clen:
            # pad mspec
            n_pad = clen - N
            mspec = F.pad(mspec, (0, 0, 0, n_pad), value=mspec.min())
        elif N > clen:
            crop_start = random.randint(0, N - clen)
            mspec = mspec[crop_start:crop_start+clen]
        return mspec

def get_loader(files, spk_embs, len_crop, batch_size=16, 
                num_workers=0, shuffle=False, scale=None, shift=None):
    """Build and return a data loader."""
    dataset = AutoVCDataset(files, spk_embs, len_crop, scale=scale, shift=shift)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  drop_last=shuffle, pin_memory=shuffle) # set pin memory to True if training.
    return data_loader



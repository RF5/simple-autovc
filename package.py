import torch
from model_vc import Generator
import torch
from hp import hp
from spec_utils import get_mspec, get_mspec_from_array, pad_seq
from pathlib import Path

class AutoVC(Generator):

    def __init__(self, dim_neck=32, dim_emb=256, dim_pre=512, freq=32, normalize=True):
        super().__init__(dim_neck, dim_emb, dim_pre, freq)
        self.norm_mel = lambda x: (x + hp.mel_shift) / hp.mel_scale
        self.denorm_mel = lambda x: (x*hp.mel_scale) - hp.mel_shift
        self.normalize = normalize

    def normalize_mel(self, x):
        r""" Normalize a mel-spectrogram for inference """
        return self.norm_mel(x) if self.normalize else x

    def denormalize_mel(self, x):
        r""" Denormalize a mel-spectrogram for vocoding """
        return self.denorm_mel(x) if self.normalize else x
    
    def mspec_from_file(self, pth):
        r""" Get a mel spectrogram from an audio file """
        mspec = get_mspec(pth, is_hifigan=True) # (N, n_mels)
        mspec = self.normalize_mel(mspec)
        return mspec

    def pad_mspec(self, mel):
        mspec_padded, len_pad = pad_seq(mel)
        if not self.normalize: raise NotImplementedError("Padding assumes spectrograms scales with min value of 0.")
        return mspec_padded, len_pad

    def mspec_from_numpy(array, sampling_rate):
        r""" Get a mel spectrogram from a numpy `array` of a waveform with a given `sampling_rate` """
        mspec = get_mspec_from_array(array, sampling_rate, is_hifigan=True, return_waveform=True) # (N, n_mels)
        mspec = self.normalize_mel(mspec)
        return mspec

def autovc(pretrained=True, progress=True, normalize=True, **kwargs):
    r""" 
    AutoVC model trained on 100 speakers from the VCTK dataset.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        normalize (bool): whether the model should by default normalize input mel-spectrograms (default True)
        kwargs: arguments passed to the spectrogram transform
    """
    model = AutoVC(normalize=normalize, **kwargs)
    if pretrained:
        state = torch.hub.load_state_dict_from_url("https://github.com/RF5/simple-autovc/releases/download/stable/checkpoint_noopt.pth", 
                                                progress=progress)
        model.load_state_dict(state['model_state_dict'])

    return model

def hifigan(pretrained=True, progress=True, **kwargs):
    r""" 
    HiFiGAN vocoder model, fine-tuned from the https://github.com/jik876/hifi-gan/ repo.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        kwargs: arguments passed to the spectrogram transform
    """
    if pretrained:
        svpath = Path(torch.hub.get_dir() + '/simple-autovc-hifigan.pt')
        if not svpath.is_file():
            torch.hub.download_url_to_file("https://github.com/RF5/simple-autovc/releases/download/stable/packaged_hifigan.pt",
                                            svpath, progress=progress)

        importer = torch.package.PackageImporter(svpath)
        vocoder = importer.load_pickle("models", "hifigan.pkl")
    if not pretrained:
        raise NotImplementedError("HiFiGAN pretrained model saved as torch package. Please use original hifigan repo to train new hifigan from scratch.")

    return vocoder
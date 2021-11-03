# simple-autovc
A simple, performant re-implementation of AutoVC trained on VCTK.

## Motivation
The [original author's repo](https://github.com/auspicious3000/AutoVC) has not released models which produce the same quality conversions as those presented in the [demo](https://auspicious3000.github.io/autovc-demo/).
In this repo I aim to get as close as possible to the demo performance and to release the model publicly for any to use. 

## Description
I use the model definition provided by the original author but use the [HiFi-GAN vocoder](https://github.com/jik876/hifi-gan/) and its associated mel-spectrogram transform.
Concretely, the sample rate is set at 16kHz as in the original model and the number of training steps is increased drastically from [that stated in the paper](https://arxiv.org/pdf/1905.05879.pdf) -- from 100k steps to 2.3 million steps. 
The speaker embedding network is also pretrained on a larger external dataset.

Otherwise, all the hyperparameters are the same as those from the paper, original author repo, or github issues of the original author repo where appropriate.
The 3 model components are as follows:
- AutoVC -- trained and loaded in this repo.
- Speaker embedding network -- obtained from the pretrained [simple speaker embedding repo](https://github.com/RF5/simple-speaker-embedding) using torch hub. 
- HiFi-GAN vocoder -- using a pretrained model obtained from the [original paper author](https://github.com/auspicious3000).

## Usage
To use the pretrained models, no dependancies aside from `pytorch`, `librosa`, `scipy`, and `numpy` are required. 
The models use [torch hub](https://pytorch.org/docs/stable/hub.html), making loading exceedingly simple:

### Quickstart

**Step 1**: load all the models

```python
import torch 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load the pretrained autovc model:
autovc = torch.hub.load('RF5/simple-autovc', 'autovc').to(device)
autovc.eval()
# Load the pretrained hifigan model:
hifigan = torch.hub.load('RF5/simple-autovc', 'hifigan').to(device)
hifigan.eval()
# Load speaker embedding model:
sse = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').to(device)
sse.eval()

```

**Step 2**: do inference on some utterances of your choice

```python
# Get mel spectrogram
mel = autovc.mspec_from_file('example/source_uttr.flac') 
# or autovc.mspec_from_numpy(numpy array, sampling rate) if you have a numpy array

# Get embedding for source speaker
sse_src_mel = sse.melspec_from_file('example/source_uttr.flac')
with torch.no_grad(): 
    src_embedding = sse(sse_src_mel[None].to(device))
# Get embedding for target speaker
sse_trg_mel = sse.melspec_from_file('example/target_uttr.flac')
with torch.no_grad(): 
    trg_embedding = sse(sse_trg_mel[None].to(device))

# Do the actual voice conversion!
with torch.no_grad():
    spec_padded, len_pad = autovc.pad_mspec(mel)
    x_src = spec_padded.to(device)[None]
    s_src = src_embedding.to(device)
    s_trg = trg_embedding.to(device)
    x_identic, x_identic_psnt, _ = autovc(x_src, s_src, s_trg)
    if len_pad == 0: x_trg = x_identic_psnt[0, 0, :, :]
    else: x_trg = x_identic_psnt[0, 0, :-len_pad, :]

# x_trg is now the converted spectrogram!
```

**Step 3**: vocode output spectrogram:

```python
# Make a vocode function
@torch.no_grad()
def vocode(spec):
    # denormalize mel-spectrogram
    spec = autovc.denormalize_mel(spec)
    _m = spec.T[None]
    waveform = hifigan(_m.to(device))[0]
    return waveform.squeeze()

converted_waveform = vocode(x_trg) # output waveform 
# Save waveform as wav file
import soundfile as sf
sf.write('converted_uttr.flac', converted_waveform.cpu().numpy(), 16000)

```

Doing this for the example utterance in the `example/` folder yields the following:

1. Source utterance:
1.1 raw 48kHz:

https://user-images.githubusercontent.com/23717819/131484933-4f70eb43-452a-44a7-8569-a83da745241f.mp4

1.2 vocoded 16kHz:

https://user-images.githubusercontent.com/23717819/131485889-ecfc0a56-6ccd-4a28-b56d-b0285677c8ad.mp4

2. Reference style utterance: 

2.1 raw 48kHz:

https://user-images.githubusercontent.com/23717819/131485048-c9ec4283-7c3f-459c-ad52-0f0cb90eb02e.mp4

2.2 vocoded 16kHz:

https://user-images.githubusercontent.com/23717819/131485960-cee191e5-56e5-403e-a0fb-00da53fc95d3.mp4

3. Converted output utterance (vocoded 16kHz): 

https://user-images.githubusercontent.com/23717819/131485109-06616224-cf2c-41ee-ab8a-8c1caba27d13.mp4

Note as well that the input or reference utterance may be speakers unseen during training, or any audio file if you are feeling very brave.

# Training

## AutoVC
To train the model, simply set the root data directory in `hp.py`, and run `train.py` with the best arguments for your use case.
Note that `train.py` is currently set up to load data in a VCTK-style folder format, so you may need to rework it to your dataset if you use a new one.

You can save time during training by pre-computing the mel-spectrograms from the waveforms using `spec_utils.py`, in which case just pass the precomputed mel-spectrogram folder to `train.py` as the appropriate argument. 

## Speaker embedding network
Please see the [details in the original repo](https://github.com/RF5/simple-speaker-embedding) if you wish to further train it, but it is pretty good and works well even on several unseen languages. 

## HiFi-GAN
Please see the instructions in [the HiFi-GAN repo](https://github.com/jik876/hifi-gan) on how to fine-tune the vocoder.
To do this, you would need to generate reconstructed AutoVC spectrogram outputs and pair them with the ground-truth waveforms. HiFi-GAN fine-tuning will then use teacher forcing to make the vocoder better adapt to AutoVC's output. 
Remember to set the sampling rate to 16kHz for this step, as the default for HiFi-GAN is 22.05kHz. 

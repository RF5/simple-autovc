import datetime
import math
import os
import random
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from fastprogress import master_bar, progress_bar
from torch.nn import utils
from torch.utils import tensorboard

from data import AutoVCDataset, get_loader
from hp import hp
from model_vc import Generator


def train(args):
    print("[BACKEND] Setting up paths and training.")
    out_path = Path(hp.output_path)
    os.makedirs(out_path, exist_ok=True) # Make output directory

    device = torch.device(hp.device)

    # For now, no precomputed spectrograms
    if args.mel_path is None:
        ds_root = Path(hp.data_root)
    else:
        print("[DATA] Using precomputed mels from ", args.mel_path)
        ds_root = Path(args.mel_path)

    # Get speaker folders (only directories with numeric names)
    spk_folders = []
    for path in sorted(list(ds_root.iterdir())):
        if path.is_dir():
            if path.parts[-1].isnumeric():
                spk_folders.append(path)
    print(f"[DATA] Found a total of {len(spk_folders)} speakers")

    # Gather training / testing paths.
    random.seed(hp.seed)
    train_spk_folders = sorted(random.sample(spk_folders, k=hp.n_train_speakers))
    test_spk_folders = sorted(list(set(spk_folders) - set(train_spk_folders)))
    train_files = []
    for pth in train_spk_folders: train_files.extend(list(pth.iterdir()))
    test_files = []
    for pth in test_spk_folders: test_files.extend(list(pth.iterdir()))
    print(f"[DATA] Split into {len(train_spk_folders)} train speakers ({len(train_files)} files)")
    print(f"[DATA] and {len(test_spk_folders)} test speakers ({len(test_files)} files)")

    # Getting embeddings:
    sse = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder').to(device)
    sse.eval()

    # mb = master_bar(spk_folders)
    spk_out_path = Path(hp.speaker_embedding_dir)
    spk_embs = {}
    os.makedirs(spk_out_path, exist_ok=True)
    print("[SPEAKER EMBEDDING] Gathering speaker embeddings")
    for spk_folder in spk_folders:
        random.seed(hp.seed)
        sample_uttrs = random.sample(list(spk_folder.iterdir()), k=hp.n_uttr_per_spk_embedding)
        embs = []
        if (spk_out_path/f"{spk_folder.stem}_sse_emb.pt").is_file(): 
            spk_embs[spk_folder.stem] = torch.load(spk_out_path/f"{spk_folder.stem}_sse_emb.pt")
            continue

        for i, uttr_pth in enumerate(sample_uttrs):
            # mb.child.comment = f"processing speaker {spk_folder.stem} ({i} of {len(sample_uttrs)})"
            mel = sse.melspec_from_file(uttr_pth).to(device)
            if str(uttr_pth).endswith('.pt'): 
                raise NotImplementedError(("If spectrograms are not precomputed, please do not use pre-computed mel-spectrograms in args."))
            with torch.no_grad():
                embedding = sse(mel[None])[0]
            embs.append(embedding.cpu())
        
        emb = torch.stack(embs, dim=0)
        emb = emb.mean(dim=0)
        spk_embs[spk_folder.stem] = emb
        torch.save(emb, spk_out_path/f"{spk_folder.stem}_sse_emb.pt")

    del sse
    torch.cuda.empty_cache()

    print("[DATA] Constructing final dataloaders")
    train_dl = get_loader(train_files, spk_embs, hp.len_crop, hp.bs, 
                        shuffle=True, shift=hp.mel_shift, scale=hp.mel_scale)
    test_dl = get_loader(test_files, spk_embs, hp.len_crop, hp.bs, 
                        shuffle=False, shift=hp.mel_shift, scale=hp.mel_scale)

    print(list(train_dl))
    for i, (x_src, s_src) in enumerate(train_dl):
        print(i, x_src, s_src)

    print("[LOGGING] Setting up logger")
    writer = tensorboard.writer.SummaryWriter(out_path)
    keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

    print("[MODEL] Setting up model")
    G = Generator(32, 256, 512, 32).to(device)
    opt = torch.optim.Adam(G.parameters(), hp.lr)
    if args.fp16: 
        print("[TRAIN] Using fp16 training.")
        scaler = GradScaler()

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        epoch = ckpt['epoch']
        ite = ckpt['iter']
        G.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['opt_state_dict'])
        print(f"[CHECKPOINT] Loaded checkpoint starting from epoch {epoch} (iter {ite})",
                f" with last known loss {ckpt['loss']:6.5f}")

    print("[TRAIN] Beginning training")
    start_time = time.time()
    running_loss = 0.0
    n_epochs = math.ceil(hp.n_iters / len(train_dl))
    iter = 0
    # mb = master_bar(range(n_epochs))
    for epoch in range(n_epochs):

        G.train()
        # pb = progress_bar(enumerate(train_dl), total=len(train_dl), parent=mb)
        for i, (x_src, s_src) in enumerate(train_dl):
            x_src = x_src.to(device)
            s_src = s_src.to(device)
            opt.zero_grad()

            # fp16 enable
            if args.fp16:
                with autocast():
                    # Identity mapping loss
                    x_identic, x_identic_psnt, code_real = G(x_src, s_src, s_src)
                    g_loss_id = F.mse_loss(x_src, x_identic.squeeze(1))   
                    g_loss_id_psnt = F.mse_loss(x_src, x_identic_psnt.squeeze(1))   
                    
                    # Code semantic loss.
                    code_reconst = G(x_identic_psnt.squeeze(1), s_src, None)
                    g_loss_cd = F.l1_loss(code_real, code_reconst)

                    g_loss = g_loss_id + hp.mu*g_loss_id_psnt + hp.lamb*g_loss_cd
                scaler.scale(g_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                # Identity mapping loss
                x_identic, x_identic_psnt, code_real = G(x_src, s_src, s_src)
                g_loss_id = F.mse_loss(x_src, x_identic.squeeze(1))   
                g_loss_id_psnt = F.mse_loss(x_src, x_identic_psnt.squeeze(1))   
                
                # Code semantic loss.
                code_reconst = G(x_identic_psnt.squeeze(1), s_src, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                g_loss = g_loss_id + hp.mu*g_loss_id_psnt + hp.lamb*g_loss_cd
                g_loss.backward()
                opt.step()

            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            
            # lerp smooth running loss
            running_loss = running_loss + 0.1*(float(g_loss) - running_loss)
            # mb.child.comment = f"loss = {float(running_loss):6.5f}"

            if iter % hp.print_log_interval == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, iter+1, hp.n_iters)
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                # mb.write(log)
            
            if iter % hp.tb_log_interval == 0:
                for tag in keys: writer.add_scalar(tag, loss[tag], iter)
                writer.add_scalar('G/loss', g_loss.item(), iter)
            
            iter += 1
            if iter >= hp.n_iters:
                print("[TRAIN] Training completed.")
                break
        
        # mb.write(f"[TRAIN] epoch {epoch} completed. Beginning eval.")
        G.eval()
        # pb = progress_bar(enumerate(test_dl), total=len(test_dl), parent=mb)
        valid_losses = {tag: [] for tag in keys}
        valid_losses['G/loss'] = []
        for i, (x_src, s_src) in enumerate(test_dl):
            x_src = x_src.to(device)
            s_src = s_src.to(device)

            with torch.no_grad():
                # Identity mapping loss
                x_identic, x_identic_psnt, code_real = G(x_src, s_src, s_src)
                g_loss_id = F.mse_loss(x_src, x_identic.squeeze(1))   
                g_loss_id_psnt = F.mse_loss(x_src, x_identic_psnt.squeeze(1))   
                
                # Code semantic loss.
                code_reconst = G(x_identic_psnt.squeeze(1), s_src, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                g_loss = g_loss_id + hp.mu*g_loss_id_psnt + hp.lamb*g_loss_cd

            valid_losses['G/loss_id'].append(g_loss_id.item())
            valid_losses['G/loss_id_psnt'].append(g_loss_id_psnt.item())
            valid_losses['G/loss_cd'].append(g_loss_cd.item())
            valid_losses['G/loss'].append(g_loss.item())
            # mb.child.comment = f"loss = {float(g_loss):6.5f}"
        
        valid_losses = {k: np.mean(valid_losses[k]) for k in valid_losses.keys()}
        for tag in valid_losses.keys(): writer.add_scalar('valid/' + tag, valid_losses[tag], iter)
        pst = [f"{k}: {valid_losses[k]:5.4f}" for k in valid_losses.keys()]
        # mb.write(f"[TRAIN] epoch {epoch} eval metrics: " + '\t'.join(pst))

        if iter >= hp.n_iters: break
    
    print("[CLEANUP] Saving model")
    torch.save({
        'epoch': epoch,
        'iter': iter,
        'model_state_dict': G.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'loss': valid_losses['G/loss']
    }, out_path/'checkpoint_last.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='autovc trainer')
    parser.add_argument('--checkpoint', action='store', required=False, default=None,
                        help='checkpoint to restore from')
    parser.add_argument('--fp16', required=False, default=False, action='store', type=bool,
                        help='use fp16 in training')
    parser.add_argument('--mel_path', required=False, default=None, action='store',
                        help='path to precomputed spectrograms. Compute them on the fly if not.')
    parser.add_argument('--lj_path', required=False, default=None,
                        help="Add LJSpeech dataset")

    args = parser.parse_args()
    train(args)

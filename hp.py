
class hp:

    data_root = '/home/baas/data-hdd/datasets/vctk'
    n_train_speakers = 100
    n_test_speakers = 10 # these are the zero-shot speakers

    # speaker embedding settings
    n_uttr_per_spk_embedding = 10
    speaker_embedding_dir = './sse_embeddings/'

    # train settings
    output_path = './outputs/run1/'
    device = 'cuda'
    len_crop = 128
    bs = 4 # 2x that of paper
    n_iters = 100000 # 100k as per paper
    lamb = 1
    mu = 1
    tb_log_interval = 10
    print_log_interval = 1000

    lr = 1e-4 # according to github issues, no lr schedule is used

    seed = 100
    mel_shift = 12
    mel_scale = 12
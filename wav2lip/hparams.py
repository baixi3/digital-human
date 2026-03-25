class HParams:
    def __init__(self, **kwargs):
        self.data = dict(kwargs)

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"'HParams' 没有这个 {key}")
        return self.data[key]

hparams = HParams(
    num_mels=80,
    use_lws=False,
    n_fft=800,
    hop_size=200,
    win_size=800,
    sample_rate=16000,
    frame_shift_ms=None,
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,
    preemphasize=True,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,
)

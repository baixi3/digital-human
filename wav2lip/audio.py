import librosa
import librosa.filters
import numpy as np
from scipy import signal

from .hparams import hparams as hp

# 对原始音频做预加重，增强高频部分，让语音特征更明显。
def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def get_hop_size():
    hop_size = hp.hop_size
    if hop_size is None:
        assert hp.frame_shift_ms is not None
        hop_size = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    return hop_size

# 把原始音频波形转换成 mel 频谱特征，供 Wav2Lip 使用。
def melspectrogram(wav):
    stft = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    mel = _amp_to_db(_linear_to_mel(np.abs(stft))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(mel)
    return mel


def _stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft,
        hop_length=get_hop_size(),
        win_length=hp.win_size,
    )


_mel_basis = None

# 把普通线性频谱映射到 mel 频谱。
def _linear_to_mel(spectogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectogram)


def _build_mel_basis():
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=float(hp.sample_rate),
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
    )


def _amp_to_db(x):
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

# 把 mel 频谱进一步缩放到模型更适合处理的数值范围。
def _normalize(spectrogram):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip(
                (2 * hp.max_abs_value)
                * ((spectrogram - hp.min_level_db) / (-hp.min_level_db))
                - hp.max_abs_value,
                -hp.max_abs_value,
                hp.max_abs_value,
            )
        return np.clip(
            hp.max_abs_value * ((spectrogram - hp.min_level_db) / (-hp.min_level_db)),
            0,
            hp.max_abs_value,
        )

    assert spectrogram.max() <= 0 and spectrogram.min() - hp.min_level_db >= 0
    if hp.symmetric_mels:
        return (
            (2 * hp.max_abs_value)
            * ((spectrogram - hp.min_level_db) / (-hp.min_level_db))
            - hp.max_abs_value
        )
    return hp.max_abs_value * ((spectrogram - hp.min_level_db) / (-hp.min_level_db))

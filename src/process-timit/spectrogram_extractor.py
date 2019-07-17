import os
import numpy as np
import soundfile as sf
import librosa

sample_rate = 16000
n_fft = int(16000 * 0.025)
win_length = n_fft
hop_length = int(16000 * 0.01)

def extract_spec(wav_path):
    # print('extract spectrogram:', wav_path)
    data, sr = sf.read(wav_path)

    D = librosa.core.stft(data, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = "hamming", center=False)

    spec, phase = librosa.magphase(D)
    spec = spec.T
    spec = np.log1p(spec)

    phase = phase.T
    phase = np.angle(phase)

    return spec, phase

def reconstruct_waveform(spec, phase):
    mag = np.expm1(spec)
    D = mag * np.exp(phase * 1j)
    D = D.T
    raw = librosa.core.istft(D, hop_length = hop_length, win_length = win_length, window = "hamming", center=False)

    return raw

def write_wav(raw, path):
    sf.write(path, raw, sample_rate)

def griffin_lim(stftm_matrix, shape, min_iter=20, max_iter=50, delta=20):
    y = np.random.random(shape)
    y_iter = []

    for i in range(max_iter):
        if i >= min_iter and (i - min_iter) % delta == 0:
            y_iter.append((y, i))
        stft_matrix = librosa.core.stft(y, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = "hamming", center=False)
        stft_matrix = stftm_matrix * stft_matrix / np.abs(stft_matrix)
        y = librosa.core.istft(stft_matrix, hop_length = hop_length, win_length = win_length, window = "hamming", center=False)
    y_iter.append((y, max_iter))

    return y_iter

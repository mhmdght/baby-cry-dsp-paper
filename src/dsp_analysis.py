import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from scipy.fftpack import dct
from scipy.special import softmax

# ==========================================
# LOAD AUDIO
# ==========================================

audio_path = "sample.wav"

signal, sr = librosa.load(
    audio_path,
    sr=16000
)

# ==========================================
# PRE EMPHASIS
# y(n)=x(n)-a*x(n-1)
# ==========================================

a = 0.97

pre_emphasis = np.append(
    signal[0],
    signal[1:] - a * signal[:-1]
)

plt.figure(figsize=(12,4))
plt.plot(signal)
plt.title("Original Signal")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(pre_emphasis)
plt.title("Pre-Emphasis Signal")
plt.show()

# ==========================================
# FFT
# ==========================================

fft_original = np.abs(
    np.fft.rfft(signal)
)

fft_pre = np.abs(
    np.fft.rfft(pre_emphasis)
)

freqs = np.fft.rfftfreq(
    len(signal),
    1/sr
)

plt.figure(figsize=(12,4))
plt.plot(freqs, fft_original)
plt.title("FFT Original")
plt.xlabel("Frequency")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(freqs, fft_pre)
plt.title("FFT After Pre-Emphasis")
plt.xlabel("Frequency")
plt.show()

# ==========================================
# HAMMING WINDOW
# ==========================================

frame_length = 512

frame = signal[:frame_length]

window = np.hamming(frame_length)

windowed = frame * window

plt.figure(figsize=(10,4))
plt.plot(window)
plt.title("Hamming Window")
plt.show()

# ==========================================
# MEL FILTER BANK
# ==========================================

mel_filter = librosa.filters.mel(
    sr=sr,
    n_fft=2048,
    n_mels=40
)

plt.figure(figsize=(12,6))

for filt in mel_filter:
    plt.plot(filt)

plt.title("Mel Filter Bank")
plt.show()

# ==========================================
# MEL SPECTROGRAM
# ==========================================

mel_spec = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_mels=40
)

plt.figure(figsize=(10,5))

librosa.display.specshow(
    librosa.power_to_db(mel_spec),
    x_axis='time',
    y_axis='mel'
)

plt.colorbar()

plt.title("Mel Spectrogram")

plt.show()

# ==========================================
# DCT
# ==========================================

log_mel = np.log(
    mel_spec + 1e-8
)

mfcc_manual = dct(
    log_mel,
    axis=0,
    norm='ortho'
)

plt.figure(figsize=(10,5))

librosa.display.specshow(
    mfcc_manual[:13],
    x_axis='time'
)

plt.colorbar()

plt.title("MFCC via DCT")

plt.show()

# ==========================================
# LIBROSA MFCC
# ==========================================

mfcc_librosa = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=13
)

plt.figure(figsize=(10,5))

librosa.display.specshow(
    mfcc_librosa,
    x_axis='time'
)

plt.colorbar()

plt.title("MFCC Librosa")

plt.show()

# ==========================================
# RELU
# ==========================================

x = np.linspace(-10,10,1000)

relu = np.maximum(
    0,
    x
)

plt.figure(figsize=(8,4))

plt.plot(x,relu)

plt.title("ReLU")

plt.grid()

plt.show()

# ==========================================
# LOGSOFTMAX
# ==========================================

z = np.array(
    [1,2,3,4,5]
)

soft = softmax(z)

logsoft = np.log(soft)

plt.figure(figsize=(8,4))

plt.bar(
    range(len(soft)),
    soft
)

plt.title("Softmax")

plt.show()

plt.figure(figsize=(8,4))

plt.bar(
    range(len(logsoft)),
    logsoft
)

plt.title("LogSoftmax")

plt.show()

print("Softmax")

print(soft)

print("\nLogSoftmax")

print(logsoft)
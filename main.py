import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# CONFIG
# =====================================

SAMPLE_RATE = 16000
TARGET_DURATION = 7
TARGET_LENGTH = SAMPLE_RATE * TARGET_DURATION

# =====================================
# PREPROCESSING
# =====================================

def load_audio(filepath):

    signal, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE
    )

    return signal


def pad_trim(signal):

    if len(signal) < TARGET_LENGTH:

        signal = np.pad(
            signal,
            (0, TARGET_LENGTH - len(signal))
        )

    else:

        signal = signal[:TARGET_LENGTH]

    return signal

# =====================================
# DATA AUGMENTATION
# =====================================

def time_shift(signal):

    shift = np.random.randint(-500, 500)

    return np.roll(signal, shift)


def time_stretch(signal):

    rate = np.random.uniform(0.8, 1.2)

    stretched = librosa.effects.time_stretch(
        signal,
        rate=rate
    )

    return pad_trim(stretched)


def pitch_shift(signal):

    shifted = librosa.effects.pitch_shift(
        signal,
        sr=SAMPLE_RATE,
        n_steps=4
    )

    return shifted


def add_white_noise(signal):

    noise = np.random.normal(
        0,
        np.sqrt(0.000025),
        len(signal)
    )

    return signal + noise


def random_slice(signal):

    slice_length = SAMPLE_RATE

    if len(signal) <= slice_length:
        return signal

    start = np.random.randint(
        0,
        len(signal) - slice_length
    )

    sliced = signal[start:start + slice_length]

    return pad_trim(sliced)

# =====================================
# MFCC EXTRACTION
# =====================================

def extract_mfcc(signal):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc,
        axis=1
    )

    return mfcc_mean


def plot_mfcc(signal, title):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=13
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        mfcc,
        x_axis='time'
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.show()

# =====================================
# WAVEFORM PLOT
# =====================================

def plot_wave(signal, title):

    plt.figure(figsize=(12,4))

    plt.plot(signal)

    plt.title(title)

    plt.tight_layout()

    plt.show()

# =====================================
# DATASET CREATION
# =====================================

def build_dataset(root_folder):

    X = []
    y = []

    classes = sorted(os.listdir(root_folder))

    print("Classes:", classes)

    for label_idx, class_name in enumerate(classes):

        class_path = os.path.join(
            root_folder,
            class_name
        )

        wav_files = glob.glob(
            os.path.join(class_path, "*.wav")
        )

        print(
            class_name,
            len(wav_files)
        )

        for wav in wav_files:

            try:

                signal = load_audio(wav)

                signal = pad_trim(signal)

                feature = extract_mfcc(signal)

                X.append(feature)

                y.append(label_idx)

            except Exception as e:

                print("ERROR:", wav)
                print(e)

    X = np.array(X)
    y = np.array(y)

    return X, y, classes

# =====================================
# VISUALIZE ONE SAMPLE
# =====================================

def visualize_augmentations(filepath):

    signal = load_audio(filepath)

    signal = pad_trim(signal)

    shifted = time_shift(signal)

    stretched = time_stretch(signal)

    pitched = pitch_shift(signal)

    noisy = add_white_noise(signal)

    sliced = random_slice(signal)

    plot_wave(signal, "Original")

    plot_wave(shifted, "Time Shift")

    plot_wave(stretched, "Time Stretch")

    plot_wave(pitched, "Pitch Shift")

    plot_wave(noisy, "White Noise")

    plot_wave(sliced, "Slice")

    plot_mfcc(signal, "MFCC - Original")

    plot_mfcc(pitched, "MFCC - Pitch Shift")

# =====================================
# MAIN
# =====================================

if __name__ == "__main__":

    DATASET_PATH = "data"

    # -----------------------------
    # Dataset Build
    # -----------------------------

    X, y, classes = build_dataset(
        DATASET_PATH
    )

    print("\nFeature Matrix Shape:")
    print(X.shape)

    print("\nLabels Shape:")
    print(y.shape)

    # -----------------------------
    # Class Distribution
    # -----------------------------

    plt.figure(figsize=(8,5))

    sns.countplot(x=y)

    plt.title("Class Distribution")

    plt.show()

    # -----------------------------
    # Show One Example
    # -----------------------------

    first_class = classes[0]

    sample_file = glob.glob(
        os.path.join(
            DATASET_PATH,
            first_class,
            "*.wav"
        )
    )[0]

    visualize_augmentations(
        sample_file
    )
# =====================================================
# 🍼 Baby Cry Classification - Jupyter Notebook
# Based on: Ozcan & Gungor (2025)
# =====================================================

# =====================================================
# CELL 1: Install Required Libraries (Run Once)
# =====================================================

 !pip install librosa numpy pandas matplotlib seaborn scikit-learn tensorflow scikeras scipy

# =====================================================
# CELL 2: Imports & Configuration
# =====================================================

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from scikeras.wrappers import KerasClassifier

warnings.filterwarnings('ignore')

# =====================================================
# CELL 3: Configuration (Based on Paper)
# =====================================================

class Config:
    # Audio parameters
    SAMPLE_RATE = 16000
    TARGET_DURATION = 7
    N_MFCC = 13
    FRAME_LENGTH = 512
    HOP_LENGTH = 256
    
    # Data split (75% train, 10% val, 15% test)
    TRAIN_RATIO = 0.75
    VAL_RATIO = 0.10
    TEST_RATIO = 0.15
    
    # Model architecture
    DENSE_LAYERS = 5
    DROPOUT_RATE = 0.5
    ACTIVATION = 'relu'
    
    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 100
    OPTIMIZER = 'adam'
    
    # Paths
    DATASET_PATH = "data"  # Change this to your dataset path
    OUTPUT_PATH = "output"
    MODELS_PATH = "models"
    
    # Classes (from Donate a Cry dataset)
    CLASSES = ["Belly pain", "Burping", "Discomfort", "Hungry", "Tired"]
    
    @classmethod
    def create_dirs(cls):
        for path in [cls.OUTPUT_PATH, cls.MODELS_PATH]:
            os.makedirs(path, exist_ok=True)
    
    @classmethod
    def get_target_length(cls):
        return cls.SAMPLE_RATE * cls.TARGET_DURATION

# Create directories
Config.create_dirs()

print("✅ Configuration loaded!")
print(f"Classes: {Config.CLASSES}")
print(f"Target length: {Config.get_target_length()} samples")

# =====================================================
# CELL 4: Data Augmentation (Section 2.3.1)
# =====================================================

class DataAugmentation:
    """Data augmentation techniques from paper Table 3"""
    
    @staticmethod
    def time_shift(signal, max_shift=500):
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)
    
    @staticmethod
    def time_stretch(signal, min_rate=0.8, max_rate=1.2):
        rate = np.random.uniform(min_rate, max_rate)
        stretched = librosa.effects.time_stretch(signal, rate=rate)
        return DataAugmentation._pad_trim(stretched)
    
    @staticmethod
    def pitch_shift(signal, n_steps=4, sr=16000):
        return librosa.effects.pitch_shift(signal, sr=sr, n_steps=n_steps)
    
    @staticmethod
    def add_white_noise(signal, variance=0.000025):
        noise = np.random.normal(0, np.sqrt(variance), len(signal))
        return signal + noise
    
    @staticmethod
    def random_slice(signal, slice_duration=1.0, sr=16000):
        slice_length = int(slice_duration * sr)
        if len(signal) <= slice_length:
            return signal
        start = np.random.randint(0, len(signal) - slice_length)
        sliced = signal[start:start + slice_length]
        return DataAugmentation._pad_trim(sliced)
    
    @staticmethod
    def _pad_trim(signal, target_len=None):
        if target_len is None:
            target_len = Config.get_target_length()
        if len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)))
        else:
            signal = signal[:target_len]
        return signal
    
    @staticmethod
    def apply_all(signal):
        """Apply all augmentations"""
        return [
            signal,
            DataAugmentation.time_shift(signal),
            DataAugmentation.time_stretch(signal),
            DataAugmentation.pitch_shift(signal),
            DataAugmentation.add_white_noise(signal),
            DataAugmentation.random_slice(signal)
        ]

print("✅ Data augmentation loaded!")

# =====================================================
# CELL 5: Feature Extraction (Section 2.3.3)
# =====================================================

def extract_mfcc(signal, sr=16000, n_mfcc=13):
    """Extract MFCC features with mean pooling"""
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=Config.FRAME_LENGTH,
        hop_length=Config.HOP_LENGTH
    )
    return np.mean(mfcc, axis=1)

def extract_mfcc_full(signal, sr=16000, n_mfcc=13):
    """Extract full MFCC matrix for visualization"""
    return librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=Config.FRAME_LENGTH,
        hop_length=Config.HOP_LENGTH
    )

print("✅ Feature extraction loaded!")

# =====================================================
# CELL 6: Dataset Loader
# =====================================================

class DatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.classes = Config.CLASSES
        self.target_length = Config.get_target_length()
    
    def load_audio(self, filepath):
        signal, _ = librosa.load(filepath, sr=Config.SAMPLE_RATE)
        return signal
    
    def pad_trim(self, signal):
        if len(signal) < self.target_length:
            signal = np.pad(signal, (0, self.target_length - len(signal)))
        else:
            signal = signal[:self.target_length]
        return signal
    
    def build_dataset(self, augment=False):
        X, y = []
        print("\n📂 Building dataset...")
        
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.dataset_path, class_name)
            wav_files = glob.glob(os.path.join(class_path, "*.wav"))
            print(f"  {class_name}: {len(wav_files)} files")
            
            for wav in wav_files:
                try:
                    signal = self.load_audio(wav)
                    signal = self.pad_trim(signal)
                    
                    features = extract_mfcc(signal)
                    X.append(features)
                    y.append(label_idx)
                    
                    if augment:
                        aug_signals = DataAugmentation.apply_all(signal)
                        for aug_signal in aug_signals[1:]:
                            aug_features = extract_mfcc(aug_signal)
                            X.append(aug_features)
                            y.append(label_idx)
                except Exception as e:
                    print(f"    ERROR: {wav}")
        
        X = np.array(X)
        y = np.array(y)
        print(f"\n✅ Total: {len(X)} samples, {X.shape[1]} features")
        return X, y
    
    def get_duration_stats(self):
        """Get duration statistics (Table 4)"""
        results = []
        for class_name in self.classes:
            durations = []
            class_path = os.path.join(self.dataset_path, class_name)
            wav_files = glob.glob(os.path.join(class_path, "*.wav"))
            for wav in wav_files:
                signal, sr = librosa.load(wav, sr=None)
                durations.append(len(signal) / sr)
            results.append({
                "Class": class_name,
                "Min": round(min(durations), 2),
                "Max": round(max(durations), 2),
                "Avg": round(sum(durations)/len(durations), 2),
                "Count": len(durations)
            })
        return pd.DataFrame(results)

print("✅ Dataset loader ready!")

# =====================================================
# CELL 7: Model Architecture (Figure 4)
# =====================================================

def create_model(optimizer='adam', dropout_rate=0.5, input_shape=(13,), num_classes=5):
    """Structure-Tuned ANN from paper"""
    model = Sequential([
        # Layer 1: 128 units
        Dense(128, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 2: 128 units
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 3: 64 units
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 4: 64 units
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Layer 5: 32 units
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        # Output
        Dense(num_classes, activation='softmax')
    ])
    
    # Optimizer
    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    else:
        opt = SGD()
    
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Display model summary
model_demo = create_model()
model_demo.summary()

print("\n✅ Model architecture ready!")

# =====================================================
# CELL 8: Load Dataset
# =====================================================

# IMPORTANT: Change this path to your dataset location
DATASET_PATH = "data"  # <-- CHANGE THIS

loader = DatasetLoader(DATASET_PATH)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"⚠️ Dataset not found at: {DATASET_PATH}")
    print("Please create the following structure:")
    print("data/")
    print("  ├── Belly pain/")
    print("  ├── Burping/")
    print("  ├── Discomfort/")
    print("  ├── Hungry/")
    print("  └── Tired/")
else:
    # Duration stats (Table 4)
    print("📊 Duration Statistics:")
    duration_df = loader.get_duration_stats()
    display(duration_df)
    
    # Build dataset
    X, y = loader.build_dataset(augment=False)
    
    # Save
    np.save("output/X.npy", X)
    np.save("output/y.npy", y)
    
    # Class distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x=y)
    plt.title("Class Distribution (Table 2)")
    plt.xticks(range(len(Config.CLASSES)), Config.CLASSES, rotation=45)
    plt.show()

# =====================================================
# CELL 9: Train-Test Split (75% - 10% - 15%)
# =====================================================

# First split: 75% train, 25% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Second split: 10% val, 15% test (from 25% temp)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.60, stratify=y_temp, random_state=42
)

print(f"📊 Data Split:")
print(f"  Training:   {len(X_train):4d} ({len(X_train)/len(X)*100:5.1f}%)")
print(f"  Validation: {len(X_val):4d} ({len(X_val)/len(X)*100:5.1f}%)")
print(f"  Test:       {len(X_test):4d} ({len(X_test)/len(X)*100:5.1f}%)")
print(f"  Total:      {len(X):4d}")

# =====================================================
# CELL 10: Train Model
# =====================================================

# Create model
model = create_model(optimizer='adam', dropout_rate=0.5)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

# Train
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Save
model.save("models/baby_cry_classifier.h5")
with open("models/history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Model saved!")

# =====================================================
# CELL 11: Learning Curves (Figure 6)
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Training')
axes[0].plot(history.history['val_accuracy'], label='Validation')
axes[0].set_title('Learning Curves - Accuracy (Figure 6)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Training')
axes[1].plot(history.history['val_loss'], label='Validation')
axes[1].set_title('Learning Curves - Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =====================================================
# CELL 12: Evaluate on Test Set
# =====================================================

# Predict
y_pred_proba = model.predict(X_test)
y_pred = np.argmax(y_pred_proba, axis=1)

# Metrics
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='weighted')

print("📊 Test Results:")
print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  F1-Score: {test_f1:.4f}")
print(f"  Paper reported: 90% accuracy, 90% F1-Score")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=Config.CLASSES))

# =====================================================
# CELL 13: Confusion Matrix (Figure 5)
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=Config.CLASSES,
    yticklabels=Config.CLASSES
)
plt.title('Confusion Matrix (Figure 5)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# =====================================================
# CELL 14: ROC Curves (Figure 7)
# =====================================================

from sklearn.preprocessing import label_binarize

y_bin = label_binarize(y_test, classes=np.arange(len(Config.CLASSES)))

plt.figure(figsize=(8, 6))

for i in range(len(Config.CLASSES)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{Config.CLASSES[i]} (AUC={roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - One-vs-Rest (Figure 7)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# CELL 15: 10-Fold Cross Validation
# =====================================================

print("🔄 Running 10-Fold Cross Validation...")

model_cv = KerasClassifier(
    model=create_model,
    optimizer='adam',
    dropout_rate=0.5,
    batch_size=32,
    epochs=100,
    verbose=0
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    
    model_cv.fit(X_train_fold, y_train_fold)
    score = model_cv.score(X_test_fold, y_test_fold)
    cv_scores.append(score)
    print(f"  Fold {fold:2d}: {score:.4f}")

print(f"\n10-Fold CV Accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"Paper reported: 87%")

# =====================================================
# CELL 16: Summary
# =====================================================

print("=" * 60)
print("📊 FINAL SUMMARY")
print("=" * 60)
print(f"Dataset:          Donate a Cry")
print(f"Classes:          {len(Config.CLASSES)}")
print(f"Total Samples:    {len(X)}")
print(f"MFCC Features:    {Config.N_MFCC}")
print(f"Test Accuracy:    {test_acc*100:.2f}%")
print(f"Test F1-Score:    {test_f1:.4f}")
print(f"10-Fold CV:       {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
print(f"Paper Accuracy:   90%")
print(f"Paper 10-Fold CV: 87%")
print("=" * 60)
print("✅ Complete!")
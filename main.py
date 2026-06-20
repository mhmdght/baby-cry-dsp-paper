# =====================================================
# 🍼 Baby Cry Detection System - Based on Ozcan & Gungor (2025)
# Real-time cry detection using Structure-Tuned ANN
# =====================================================

import os
import numpy as np
import sounddevice as sd
import librosa
import pickle
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from collections import deque

warnings.filterwarnings('ignore')

# =====================================================
# CONFIGURATION (Based on paper)
# =====================================================

class Config:
    # Audio parameters (Section 2.3.3)
    SAMPLE_RATE = 16000
    TARGET_DURATION = 7
    N_MFCC = 13
    FRAME_LENGTH = 512
    HOP_LENGTH = 256
    
    # Model parameters (Table 5)
    DENSE_LAYERS = 5
    DROPOUT_RATE = 0.5
    BATCH_SIZE = 32
    EPOCHS = 100
    OPTIMIZER = 'adam'
    
    # Classes (from Donate a Cry dataset)
    CLASSES = ["Belly pain", "Burping", "Discomfort", "Hungry", "Tired"]
    
    # Detection
    RECORD_DURATION = 3  # seconds for real-time detection
    CONFIDENCE_THRESHOLD = 0.6
    
    @classmethod
    def get_target_length(cls):
        return cls.SAMPLE_RATE * cls.TARGET_DURATION

# =====================================================
# FEATURE EXTRACTION (Section 2.3.3)
# =====================================================

def extract_features(signal, sr=16000):
    """
    Extract MFCC features with mean pooling (as in paper)
    Returns: 13 MFCC coefficients averaged over time
    """
    # Ensure correct length (padding/trimming)
    target_len = Config.get_target_length()
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=Config.N_MFCC,
        n_fft=Config.FRAME_LENGTH,
        hop_length=Config.HOP_LENGTH
    )
    
    # Mean pooling (as in paper)
    return np.mean(mfcc, axis=1)

def extract_features_live(signal, sr=16000):
    """
    Extract features for live streaming (shorter duration)
    """
    # Pad/trim to target length
    target_len = Config.SAMPLE_RATE * Config.RECORD_DURATION
    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)))
    else:
        signal = signal[:target_len]
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=Config.N_MFCC,
        n_fft=Config.FRAME_LENGTH,
        hop_length=Config.HOP_LENGTH
    )
    
    return np.mean(mfcc, axis=1)

# =====================================================
# MODEL ARCHITECTURE (Figure 4)
# =====================================================

def create_model():
    """
    Structure-Tuned ANN from paper (Section 2.4)
    """
    model = Sequential([
        # Layer 1: 128 units
        Dense(128, activation='relu', input_shape=(Config.N_MFCC,)),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        
        # Layer 2: 128 units
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        
        # Layer 3: 64 units
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        
        # Layer 4: 64 units
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        
        # Layer 5: 32 units
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(Config.DROPOUT_RATE),
        
        # Output: 5 classes
        Dense(len(Config.CLASSES), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# =====================================================
# TRAINING FUNCTION (Optional - if model doesn't exist)
# =====================================================

def train_model(X, y, model_path='models/baby_cry_classifier.h5'):
    """
    Train the model on the Donate a Cry dataset
    """
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    print("🚀 Training ANN model...")
    
    # Split data (75% train, 10% val, 15% test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.60, stratify=y_temp, random_state=42
    )
    
    # Create model
    model = create_model()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model, history

# =====================================================
# REAL-TIME DETECTION
# =====================================================

class BabyCryDetector:
    def __init__(self, model_path='models/baby_cry_classifier.h5'):
        """
        Initialize the cry detector with trained model
        """
        self.model_path = model_path
        self.model = None
        self.classes = Config.CLASSES
        self.sample_rate = Config.SAMPLE_RATE
        self.record_duration = Config.RECORD_DURATION
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        
        # Load model
        self.load_model()
        
        # For live visualization
        self.audio_buffer = deque(maxlen=100)
        self.prediction_history = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)
        
    def load_model(self):
        """
        Load trained model from file
        """
        if os.path.exists(self.model_path):
            print(f"📂 Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model not found! Please train the model first.")
            print("   Run: python train_model.py")
            self.model = None
    
    def predict_signal(self, signal):
        """
        Predict cry type from audio signal
        """
        if self.model is None:
            return None, None
        
        # Extract features
        features = extract_features(signal, self.sample_rate)
        features = features.reshape(1, -1)
        
        # Predict
        predictions = self.model.predict(features, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class name
        class_name = self.classes[predicted_class] if predicted_class < len(self.classes) else "Unknown"
        
        return class_name, confidence, predictions[0]
    
    def record_and_predict(self, duration=None):
        """
        Record audio and predict in real-time
        """
        if duration is None:
            duration = self.record_duration
        
        print(f"🎙️ Recording for {duration} seconds...")
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Convert to mono
        signal = recording.flatten()
        
        # Predict
        class_name, confidence, all_probs = self.predict_signal(signal)
        
        return signal, class_name, confidence, all_probs
    
    def continuous_detection(self, duration=10):
        """
        Continuously detect cry sounds
        """
        print("🔴 Starting continuous detection... (Press Ctrl+C to stop)")
        print("="*60)
        
        try:
            while True:
                # Record and predict
                signal, class_name, confidence, all_probs = self.record_and_predict(duration=self.record_duration)
                
                # Display result
                if class_name and confidence >= self.confidence_threshold:
                    status = "🚨 CRY DETECTED" if confidence > 0.8 else "⚠️ CRY SUSPECTED"
                    print(f"{status}")
                    print(f"   Type: {class_name}")
                    print(f"   Confidence: {confidence*100:.1f}%")
                    print(f"   Probabilities:")
                    for i, (cls, prob) in enumerate(zip(self.classes, all_probs)):
                        bar = '█' * int(prob * 40)
                        print(f"     {i+1}. {cls:10s}: {prob*100:5.1f}% {bar}")
                else:
                    print(f"✅ Baby Calm (Confidence: {confidence*100:.1f}%)")
                
                print("-"*60)
                
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped.")

# =====================================================
# VISUALIZATION
# =====================================================

class LiveVisualizer:
    def __init__(self, detector):
        self.detector = detector
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.patch.set_facecolor('#fff0f5')
        
        # Setup axes
        self.setup_plots()
        
        # Data buffers
        self.signal_buffer = deque(maxlen=Config.SAMPLE_RATE * 3)
        self.prediction_buffer = deque(maxlen=20)
        
    def setup_plots(self):
        # Audio signal plot
        self.ax1.set_title('🎵 Audio Signal (Real-time)', fontsize=14, fontweight='bold')
        self.ax1.set_xlabel('Time (samples)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_facecolor('#ffffff')
        self.ax1.grid(True, alpha=0.3)
        
        # Prediction probabilities
        self.ax2.set_title('📊 Prediction Probabilities', fontsize=14, fontweight='bold')
        self.ax2.set_xlabel('Class')
        self.ax2.set_ylabel('Probability')
        self.ax2.set_facecolor('#ffffff')
        self.ax2.set_xticks(range(len(Config.CLASSES)))
        self.ax2.set_xticklabels(Config.CLASSES, rotation=45, ha='right')
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, alpha=0.3)
        
        # Confidence history
        self.ax3.set_title('📈 Detection Confidence History', fontsize=14, fontweight='bold')
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Confidence')
        self.ax3.set_ylim(0, 1)
        self.ax3.set_facecolor('#ffffff')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.axhline(y=Config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', label='Threshold')
        self.ax3.legend()
        
        plt.tight_layout()
    
    def update(self, frame):
        try:
            # Record and predict
            signal, class_name, confidence, all_probs = self.detector.record_and_predict()
            
            # Update buffers
            self.signal_buffer.extend(signal[:min(len(signal), 1000)])
            self.prediction_buffer.append(confidence)
            
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Plot audio signal
            if len(self.signal_buffer) > 0:
                self.ax1.plot(list(self.signal_buffer), color='#ff69b4', linewidth=1)
                self.ax1.set_title('🎵 Audio Signal (Real-time)', fontsize=14, fontweight='bold')
                self.ax1.set_xlabel('Time (samples)')
                self.ax1.set_ylabel('Amplitude')
                self.ax1.set_facecolor('#ffffff')
                self.ax1.grid(True, alpha=0.3)
            
            # Plot prediction probabilities
            colors = ['#ff6b6b' if i == np.argmax(all_probs) else '#4ecdc4' for i in range(len(all_probs))]
            self.ax2.bar(Config.CLASSES, all_probs, color=colors, alpha=0.7)
            self.ax2.set_title(f'📊 Prediction: {class_name} ({confidence*100:.1f}%)', fontsize=14, fontweight='bold')
            self.ax2.set_xlabel('Class')
            self.ax2.set_ylabel('Probability')
            self.ax2.set_facecolor('#ffffff')
            self.ax2.set_xticklabels(Config.CLASSES, rotation=45, ha='right')
            self.ax2.set_ylim(0, 1)
            self.ax2.grid(True, alpha=0.3)
            
            # Plot confidence history
            if len(self.prediction_buffer) > 0:
                self.ax3.plot(list(self.prediction_buffer), color='#6c5ce7', linewidth=2)
                self.ax3.fill_between(range(len(self.prediction_buffer)), 
                                     list(self.prediction_buffer), 
                                     alpha=0.3, color='#6c5ce7')
                self.ax3.set_title('📈 Detection Confidence History', fontsize=14, fontweight='bold')
                self.ax3.set_xlabel('Time (seconds)')
                self.ax3.set_ylabel('Confidence')
                self.ax3.set_ylim(0, 1)
                self.ax3.set_facecolor('#ffffff')
                self.ax3.grid(True, alpha=0.3)
                self.ax3.axhline(y=Config.CONFIDENCE_THRESHOLD, color='red', linestyle='--', 
                               label=f'Threshold ({Config.CONFIDENCE_THRESHOLD})')
                self.ax3.legend()
            
            plt.tight_layout()
            
            # Display prediction
            status = "🔴 CRY" if confidence > Config.CONFIDENCE_THRESHOLD else "🟢 CALM"
            print(f"{status} | {class_name} | {confidence*100:.1f}%")
            
        except Exception as e:
            print(f"Error in update: {e}")
        
        return self.ax1, self.ax2, self.ax3

# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    print("="*60)
    print("🍼 Baby Cry Detection System")
    print("Based on Ozcan & Gungor (2025)")
    print("="*60)
    
    # Initialize detector
    detector = BabyCryDetector('models/baby_cry_classifier.h5')
    
    if detector.model is None:
        print("\n⚠️ No trained model found!")
        print("Would you like to train the model? (y/n)")
        choice = input().lower()
        
        if choice == 'y':
            print("📂 Loading dataset...")
            # Load dataset (you need to have the data prepared)
            # This is a placeholder - you need to load your actual data
            print("⚠️ Please prepare the Donate a Cry dataset first.")
            print("   Run: python prepare_dataset.py")
            return
    
    # Choose mode
    print("\nSelect mode:")
    print("  1. Single Prediction (Record once)")
    print("  2. Continuous Detection (Real-time)")
    print("  3. Live Visualization (With plots)")
    print("  4. Test with sample file")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        # Single prediction
        signal, class_name, confidence, all_probs = detector.record_and_predict()
        print(f"\n📊 Result:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  All probabilities:")
        for cls, prob in zip(Config.CLASSES, all_probs):
            print(f"    {cls}: {prob*100:.1f}%")
    
    elif choice == '2':
        # Continuous detection
        detector.continuous_detection()
    
    elif choice == '3':
        # Live visualization
        if detector.model is None:
            print("❌ No model loaded!")
            return
        
        visualizer = LiveVisualizer(detector)
        ani = FuncAnimation(visualizer.fig, visualizer.update, interval=1000, cache_frame_data=False)
        plt.show()
    
    elif choice == '4':
        # Test with sample file
        print("Enter path to audio file:")
        filepath = input().strip()
        
        if os.path.exists(filepath):
            signal, _ = librosa.load(filepath, sr=Config.SAMPLE_RATE)
            class_name, confidence, all_probs = detector.predict_signal(signal)
            print(f"\n📊 Result:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence*100:.1f}%")
        else:
            print("❌ File not found!")

# =====================================================
# DATA PREPARATION SCRIPT (Optional)
# =====================================================

def prepare_dataset():
    """
    Prepare the Donate a Cry dataset for training
    """
    print("📂 Preparing Donate a Cry dataset...")
    
    # This is a placeholder - you need to implement dataset loading
    # based on the paper's methodology
    
    print("""
    Please download the dataset from:
    https://github.com/gveres/donateacry-corpus
    
    Expected structure:
    data/
      ├── Belly pain/
      │   └── *.wav
      ├── Burping/
      │   └── *.wav
      ├── Discomfort/
      │   └── *.wav
      ├── Hungry/
      │   └── *.wav
      └── Tired/
          └── *.wav
    """)

# =====================================================
# RUN APPLICATION
# =====================================================

if __name__ == "__main__":
    main()
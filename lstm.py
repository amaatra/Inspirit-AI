import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import librosa

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Masking, Bidirectional, Dropout # type: ignore

# 1 – LOAD AND PREPROCESS AUDIO FILES
files_fullname = sorted(f for f in glob.glob('piano_triads/*.wav') if "maj" in f or "min" in f) # to skip over dim/aug chords in the dataset

chroma_frames = []
frames_per_file = []
maj_min_classn = []
scaler = MinMaxScaler(feature_range=(0, 1))
sr_target = 22050
hop_length = 1024

def extract_chroma_features(filepath, sr_target, hop_length):
        y, sr = librosa.load(filepath, sr=sr_target, mono=True)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length).T
        return chroma

for filename in files_fullname:
    if "maj" in filename:
        maj_min_classn.append(0)
    elif "min" in filename:
        maj_min_classn.append(1)
    else:
        continue # prevent dim/aug chords from being counted
    
    chroma = extract_chroma_features(filename, sr_target, hop_length)
    chroma_frames.append(chroma)
    frames_per_file.append(len(chroma))

chroma_frames = [scaler.fit_transform(seq) for seq in chroma_frames] # seq = one element in the chroma_frames array = one "sequence" of chroma vectors...

chroma_frames_pad = pad_sequences(chroma_frames, padding="post", dtype="float32")

# 2 – SPLITTING INTO TRAIN/VALIDATION/TEST

chroma_frames_pad = np.array(chroma_frames_pad)
maj_min_classn = np.array(maj_min_classn)

X_train, X_temp, y_train, y_temp = train_test_split(chroma_frames_pad, maj_min_classn, test_size=0.25, random_state=42, stratify=maj_min_classn) # Splitting into train/temp

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # Splitting temp into validation/test

# 3 – BUILDING THE LSTM

model = Sequential([
    Masking(mask_value=0.0, input_shape=(chroma_frames_pad.shape[1], 12)),
    LSTM(32),
    Dense(16, activation="sigmoid"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4 – TRAINING THE MODEL

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=8
)

# 5 – EVALUATIING THE MODEL

val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation accuracy:", val_acc)

y_prob = model.predict(X_val)
y_pred = (y_prob <= 0.5).astype(int).flatten()

def show_confusion_matrix():
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Major", "Minor"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (LSTM)")
    plt.show()

show_cm = input("Show confusion matrix? y/n")

if (show_cm == "y"):
    show_confusion_matrix()
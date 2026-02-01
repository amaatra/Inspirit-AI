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
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Conv2D, InputLayer
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from tensorflow.keras.utils import to_categorical

# 1 – LOAD AND PREPROCESS AUDIO FILES

files_fullname = sorted(f for f in glob.glob('piano_triads/*.wav') if "maj" in f or "min" in f) # to skip over dim/aug chords in the dataset

chroma_frames = []
frames_per_file = []
maj_min_classn = []
scaler = MinMaxScaler(feature_range=(0, 1))
sr_target = 22050
hop_length = 1024
EPOCHS = 70
BATCHSIZE = 8

def extract_chroma_features(filepath, sr_target, hop_length):
        y, sr = librosa.load(filepath, sr=sr_target, mono=True)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
        return chroma

for filename in files_fullname:
    if "maj" in filename:
        maj_min_classn.append(0)
    elif "min" in filename:
        maj_min_classn.append(1)
    else:
        continue # prevent dim/aug chords from being counted
    
    chroma = extract_chroma_features(filename, sr_target, hop_length)
    # chroma is a two dimensional array with shape (65, 12). 65 refers to the number of frames per file and 12 refers to the number of dimensions in each frame.
    chroma_frames.append(chroma)
    frames_per_file.append(len(chroma))

print(len(chroma_frames))

# chroma_frames_pad = pad_sequences(chroma_frames, padding="post", dtype="float32")
# chroma_frames_pad is a 3-dimensional array with shape (288, 12, 65). 288 refers to the number of files which have been converted to chroma frames, 65 refers to
# the number of frames per file and 12 refers to the number of dimensions in each frame

# 2 – SPLITTING INTO TRAIN/VALIDATION/TEST

chroma_frames = np.array(chroma_frames)
maj_min_classn = np.array(maj_min_classn)

X_train, X_temp, y_train, y_temp = train_test_split(chroma_frames, maj_min_classn, test_size=0.2, random_state=42, stratify=maj_min_classn) # Splitting into train/temp

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # Splitting temp into validation/test

X_train_reshaped = np.vstack(X_train)
scaler.fit(X_train_reshaped)

X_train = np.array([scaler.transform(x) for x in X_train])
X_val   = np.array([scaler.transform(x) for x in X_val])
X_test  = np.array([scaler.transform(x) for x in X_test])

# 3 – BUILDING THE CNN – NOT DONE YET

model = Sequential()

model.add(InputLayer(input_shape=(12,65)))
model.add(Reshape((12, 65, 1)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.1))

""" model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(Dropout(0.5)) """

# Not doing the full VGG16 model because the pooling may collapse information too aggressively...!

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

opt = keras.optimizers.RMSprop(learning_rate=0.0001)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.summary()

# 4 – TRAINING THE MODEL

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCHSIZE)

model.summary()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.legend(['train', 'validation', 'train_loss', 'val_loss'], loc='upper left')
plt.show()

# 5 – EVALUATIING THE MODEL

# val_loss, val_acc = model.evaluate(X_val, y_val)
# print("Validation accuracy:", val_acc)

y_prob = model.predict(X_val)
y_pred = np.argmax(y_prob, axis=1)

def show_confusion_matrix():
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Major", "Minor"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (CNN)")
    plt.show()

show_cm = input("Show confusion matrix? y/n")

if (show_cm == "y"):
    show_confusion_matrix()
import librosa # Audio processing
import numpy as np
import pathlib # So that the audio's filename can be read
import glob # To find all the wav files
from sklearn.preprocessing import MinMaxScaler # For normalisation
from sklearn.model_selection import train_test_split # For the train/test split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# 1 – ADDING LABELS TO AUDIO FILES

files_fullname = sorted(f for f in glob.glob('piano_triads/*.wav') if "maj" in f or "min" in f)

label_to_id = {"Major": 0, "Minor": 1}

chroma_frames = []
frames_per_file = []
labels_per_frame = []

scaler = MinMaxScaler(feature_range=(0, 1))
sr_target = 22050
hop_length = 512

for file in files_fullname:
    y, sr = librosa.load(file, sr=sr_target, mono=True)

    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length).T # Extract chroma feature (one 12D vector per frame)
    chroma_frames.append(chroma)
    frames_per_file.append(len(chroma))

    label = label_to_id["Major"] if "maj" in file else label_to_id["Minor"]
    labels_per_frame.extend([label] * len(chroma))

all_chroma_frames = np.vstack(chroma_frames)

# 3 – NORMALISING
all_chroma_frames = scaler.fit_transform(all_chroma_frames)
labels_per_frame = np.array(labels_per_frame)

# 4 RANDOM FOREST MODEL

start = time.time()

X_train, X_temp, y_train, y_temp = train_test_split(all_chroma_frames, labels_per_frame, test_size=0.2) 

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)


unique_train, counts_train = np.unique(y_train, return_counts=True) # Faster way would be to take the sum
unique_test, counts_test = np.unique(y_val, return_counts=True)

model = RandomForestClassifier(n_estimators=200, max_depth = 5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

end = time.time()
print("Runtime:", (end - start))

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy: ", accuracy)

# 5 – TESTING WITH UNSEEN DATA

test_file = "M.mp3"
test_file1 = "m1.mp3"

def predict_chord_from_file(filepath, model, scaler, sr_target=22050, hop_length=512):

    y, sr = librosa.load(filepath, sr=sr_target, mono=True)
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length).T

    chroma_scaled = scaler.transform(chroma)
    
    frame_predictions = model.predict(chroma_scaled)
    
    final_prediction = np.bincount(frame_predictions).argmax()
    
    id_to_label = {0: "Major", 1: "Minor"}
    return id_to_label[final_prediction]

test_pred1 = predict_chord_from_file(test_file, model, scaler, sr_target, hop_length)
print(test_pred1)
test_pred2 = predict_chord_from_file(test_file1, model, scaler, sr_target, hop_length)
print(test_pred2)

def show_confusion_matrix():
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Major", "Minor"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

show_cm = input("Show confusion matrix? y/n ")

if (show_cm == "y"):
    show_confusion_matrix()

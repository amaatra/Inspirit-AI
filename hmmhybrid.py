import librosa
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from hmmlearn import hmm
import matplotlib.pyplot as plt
import time

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
        maj_min_classn.append("Major")
    elif "min" in filename:
        maj_min_classn.append("Minor")
    else:
        continue # prevent dim/aug chords from being counted
    
    chroma = extract_chroma_features(filename, sr_target, hop_length)
    chroma_frames.append(chroma)
    frames_per_file.append(len(chroma))

all_chroma_frames = np.vstack(chroma_frames)
all_chroma_frames = scaler.fit_transform(all_chroma_frames)

label_array_str = np.repeat(maj_min_classn, frames_per_file) # Getting the array of classes from maj_min_classn
label_array = np.array([0 if l == "Major" else 1 for l in label_array_str]) # Converting to 0/1 so the HMM can interpret it
print("Label shape:", label_array.shape)
print("Unique labels:", np.unique(label_array, return_counts=True))

# 2 – TRAIN RANDOM FOREST

start = time.time()

def predict_random_forest():
    randomforest = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    randomforest.fit(all_chroma_frames, label_array)

    rf_preds = randomforest.predict(all_chroma_frames)
    rf_acc = accuracy_score(label_array, rf_preds)
    print("Random Forest frame-level accuracy:", rf_acc)
    print("RF classes:", randomforest.classes_)

    frame_probs = randomforest.predict_proba(all_chroma_frames)
    print("Sample probabilities (first 3 rows):", frame_probs[:3])
    return randomforest, frame_probs

# 3 – TRAIN HMM ON RANDOM FOREST PROBABILITIES

randomforest, frame_probs = predict_random_forest()

model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    init_params="mc",  # manually set startprob_ and transmat_
    params="stmc"
)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.9, 0.1],
                            [0.1, 0.9]])

model.fit(frame_probs, frames_per_file)
hybrid = model.predict(frame_probs)

# 4 – EVALUATE HYBRID MODEL

accuracy = accuracy_score(label_array, hybrid)

if(accuracy < 0.5):
    label_array = np.array([1 if l == "Major" else 0 for l in label_array_str])
    hybrid_model = predict_random_forest()
    accuracy = accuracy_score(label_array, hybrid_model)

print("Hybrid HMM accuracy:", accuracy)

end = time.time()
print("Runtime:", (end - start))

# 5 – TESTING WITH UNSEEN DATA

test_file = "M.mp3"
test_file1 = "m1.mp3"

chroma_test = extract_chroma_features(test_file, sr_target, hop_length)

def predict_chord_from_file(filepath, scaler, rf_model, hmm_model, sr_target, hop_length):
    pred_start = time.time()
    chroma = extract_chroma_features(filepath, sr_target, hop_length)
    chroma_test = scaler.transform(chroma)
    frame_probs = rf_model.predict_proba(chroma_test)
    hmm_states = hmm_model.predict(frame_probs)
    final_state = np.bincount(hmm_states).argmax()
    label_map = {0: "Major", 1: "Minor"}
    pred_end = time.time()
    pred_time = pred_end - pred_start
    return label_map[final_state], pred_time

predicted_chord_type, prediction_time = predict_chord_from_file(test_file, scaler, randomforest, model, sr_target, hop_length)
print("Predicted chord type", predicted_chord_type)
print("Prediction time:", prediction_time)

predicted_chord_type1, prediction_time1 = predict_chord_from_file(test_file1, scaler, randomforest, model, sr_target, hop_length)
print("Predicted chord type", predicted_chord_type1)
print("Prediction time:", prediction_time1)

def show_confusion_matrix():
    cm = confusion_matrix(label_array, hybrid)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Major", "Minor"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Hybrid HMM)")
    plt.show()

show_cm = input("Show confusion matrix? y/n")

if (show_cm == "y"):
    show_confusion_matrix()
import librosa
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # For the train/test split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from hmmlearn import hmm
import matplotlib.pyplot as plt
import time

# 0 – ALL  FUNCTIONS

def extract_chroma_features(filepath, sr_target, hop_length):
        y, sr = librosa.load(filepath, sr=sr_target, mono=True)
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length).T
        return chroma

def train_random_forest(rf):
    X_train = np.vstack([chroma_frames[i] for i in train_files])
    X_train = scaler.transform(X_train)
    y_train = np.hstack([
        np.full(len(chroma_frames[i]),
                0 if maj_min_classn[i] == "Major" else 1)
        for i in train_files
    ])

    rf.fit(X_train, y_train)
    return X_train, y_train

def predict_chord_from_file(filepath, scaler, rf_model, hmm_model, sr_target, hop_length):
    pred_start = time.time()
    chroma = extract_chroma_features(filepath, sr_target, hop_length)
    chroma_test = scaler.transform(chroma)
    frame_probs = rf_model.predict_proba(chroma_test)
    hmm_states = hmm_model.predict(frame_probs)
    final_state = np.bincount(hmm_states).argmax()
    
    label_map = {
        state: "Major" if label == 0 else "Minor"
        for state, label in state_to_label.items()
    }

    pred_end = time.time()
    pred_time = pred_end - pred_start
    return label_map[final_state], pred_time

def show_confusion_matrix():
    cm = confusion_matrix(y_val_frames, val_states)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Major", "Minor"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Hybrid HMM)")
    plt.show()

# 1 – LOAD AND PREPROCESS AUDIO FILES
files_fullname = sorted(f for f in glob.glob('piano_triads/*.wav') if "maj" in f or "min" in f) # to skip over dim/aug chords in the dataset

chroma_frames = []
frames_per_file = []
maj_min_classn = []
scaler = MinMaxScaler(feature_range=(0, 1))
sr_target = 22050
hop_length = 1024

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

label_array_str = np.repeat(maj_min_classn, frames_per_file) # Getting the array of classes from maj_min_classn
label_array = np.array([0 if l == "Major" else 1 for l in label_array_str]) # Converting to 0/1 so the HMM can interpret it
print("Label shape:", label_array.shape)
print("Unique labels:", np.unique(label_array, return_counts=True))

# 2 – TRAIN RANDOM FOREST

file_indices = np.arange(len(chroma_frames))

train_files, temp_files = train_test_split( # size = 288
    file_indices,
    test_size=0.2,
    random_state=42,
    stratify=maj_min_classn
)

val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    random_state=42
)

train_frames = np.vstack([chroma_frames[i] for i in train_files])
scaler.fit(train_frames)

start = time.time()

randomforest = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42
)

X_train, y_train = train_random_forest(randomforest)

# 3 – TRAIN HMM ON RANDOM FOREST PROBABILITIES

model = hmm.GaussianHMM(
    n_components=2,
    covariance_type="full",
    n_iter=100,
    init_params="mc",  # manually set startprob_ and transmat_
    params="stmc",
    random_state = 42
)
model.startprob_ = np.array([0.5, 0.5])
model.transmat_ = np.array([[0.98, 0.02],
                            [0.02, 0.98]])

X_train_probs = randomforest.predict_proba(X_train)
train_lengths = [len(chroma_frames[i]) for i in train_files]

model.fit(X_train_probs, train_lengths)
train_states = model.predict(X_train_probs)

state_to_label = {}

for state in range(model.n_components):
    mask = (train_states == state)
    if np.sum(mask) == 0:
        # state never seen during training
        state_to_label[state] = 0  # default (e.g. Major)
    else:
        majority_label = np.bincount(y_train[mask]).argmax()
        state_to_label[state] = majority_label

print("Final HMM state -> label mapping:", state_to_label)

# 4 – EVALUATE HYBRID MODEL

y_val_frames = np.hstack([
    np.full(len(chroma_frames[i]),
            0 if maj_min_classn[i] == "Major" else 1)
    for i in val_files
])

X_val = np.vstack([chroma_frames[i] for i in val_files])
X_val = scaler.transform(X_val) # Preparing validation frames

val_lengths = [len(chroma_frames[i]) for i in val_files]

X_val_probs = randomforest.predict_proba(X_val) # Getting probabilities from the random forest

val_states = model.predict(X_val_probs, val_lengths) # Decoding HMM based on validation data

accuracy = accuracy_score(y_val_frames, val_states)

print("Hybrid HMM accuracy:", accuracy)

end = time.time()
print("Runtime:", (end - start))

y, sr = librosa.load("piano_triads/A_maj_4_1.wav", sr=sr_target, mono=True)
chroma_25 = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length).T
chroma = scaler.transform(chroma_25)

rf_probs_log = np.log(X_val_probs + 1e-8)  # numerical safety
hmm_states = model.predict(rf_probs_log)

# Map states to labels
hmm_labels = np.array([state_to_label[int(s)] for s in hmm_states])
# 0 = Major, 1 = Minor

# Plot
plt.figure(figsize=(10, 3))
plt.step(range(len(hmm_labels)), hmm_labels, where="post")
plt.yticks([0, 1], ["Major", "Minor"])
plt.xlabel("Frame index")
plt.title("HMM decoded chord states over time")
plt.show()

show_cm = input("Show confusion matrix? y/n")

if (show_cm == "y"):
    show_confusion_matrix()
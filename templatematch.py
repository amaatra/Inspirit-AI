import numpy as np
from numpy.linalg import norm
import librosa
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1 - MAKING THE TEMPLATES

pitches = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')

major_base = np.array((1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0))
minor_base = np.array((1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0))

major_templates = {pitches[i] + '_major': np.roll(major_base, i) for i in range(12)} # Dictionary comprehension
minor_templates = {pitches[i] + '_minor': np.roll(minor_base, i) for i in range(12)}

all_templates = {**major_templates, **minor_templates} # Combining both into one dictionary

# 2 – COMPARING AN AUDIO SIGNAL TO A TEMPLATE

def predict_chord_type(filepath, threshold=0):
    y, sr = librosa.load(filepath)
    chromagram = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    chord_name = "C Major"
    current_cosine_similarity = 0
    prev_cosine_similarity = 0

    for i in all_templates.values():
      current_cosine_similarity = np.dot(chromagram, i) / (norm(chromagram) * norm(i))
      if (current_cosine_similarity > prev_cosine_similarity):
        prev_cosine_similarity = current_cosine_similarity

        key_found = next((key for key, value in all_templates.items() if np.array_equal(value, i)), None)

        chord_name = f"{key_found}: {current_cosine_similarity}"

    if prev_cosine_similarity < 0.5:
      return "Unknown"

    if "maj" in chord_name:
        return "Major"
    elif "min" in chord_name:
        return "Minor"

# 3 – MEASURING ACCURACY

files = sorted(glob.glob("piano_triads/*.wav", recursive=True))

true_labels = []
predicted_labels = []

for filename in files:
    true_label = ""
    if "maj" in filename:
      true_label = "Major"
    elif "min" in filename:
      true_label = "Minor"

    pred_label = predict_chord_type(filename)

    if ((true_label == "Major") or (true_label == "Minor")):
      true_labels.append(true_label)
    predicted_labels.append(pred_label)

filtered = [(true, pred) for true, pred in zip(true_labels, predicted_labels)] # Filtering out the unknown values
true_labels, predicted_labels = zip(*filtered)


print("Accuracy:", accuracy_score(true_labels, predicted_labels))

# 4 – TESTING ON UNSEEN FILE

print(predict_chord_type("m1.mp3"))

cm = confusion_matrix(true_labels, predicted_labels)

disp = ConfusionMatrixDisplay(cm, display_labels=["Major", "Minor"])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
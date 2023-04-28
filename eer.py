# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import itertools
import time
import os
import itertools
import torchaudio
import torchinfo
from torchsummary import summary

#import the model
from speechbrain.pretrained import SpeakerRecognition
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

# read database excel file
df = pd.read_csv('/home/omerhatim/thesis/ok-aura-v1.0.0/dataset.tsv', header = 0, sep = '\t')
A = df['Speaker_ID'].value_counts()



unique_speaker_ids = A.index
string_to_number = {string: idx for idx, string in enumerate(unique_speaker_ids)}

# Map the 'Speaker_ID' column using the dictionary
df['Speaker_ID'] = df['Speaker_ID'].map(string_to_number)



B = df['Speaker_ID'].value_counts()

#prefix = "/home/omerhatim/audio"

prefix = "/home/omerhatim/thesis/ok-aura-v1.0.0/clips/"

# Assuming you have a DataFrame called 'df' with a column 'path' containing the file paths
audio_files = df['Filename'].tolist()

# Add the prefix to each file path and keep only .wav files
audio_files = [os.path.join(prefix, f) for f in audio_files if f.endswith(".wav")]

# Pairwise comparison of audio files and speakers
pairs = []
unique_speaker_ids = df['Speaker_ID'].unique()

for speaker_id in unique_speaker_ids:
    speaker_files = df.loc[df['Speaker_ID'] == speaker_id, 'Filename'].tolist()
    speaker_files = [os.path.join(prefix, f) for f in speaker_files if f.endswith(".wav")]
    speaker_pairs = list(itertools.combinations(speaker_files, 2))
    speaker_pairs_with_id = [((pair[0], pair[1]), (speaker_id, speaker_id)) for pair in speaker_pairs]
    pairs.extend(speaker_pairs_with_id)

scores = []
true_labels = []
speaker_files = []

A = df['Speaker_ID'].to_list()
B = df['Filename'].to_list()
speaker_files = list(zip(A, B))

pairs = []
for speaker1, file1 in speaker_files:
    for speaker2, file2 in speaker_files:
        pairs.append([prefix+file1, prefix+file2, speaker1, speaker2])

predictions = []
for pair in pairs:
    file1, file2, speaker1, speaker2 = pair
    print(f"{speaker1} and {speaker2}")
    score, prediction = verification.verify_files(os.path.join(file1), os.path.join(file2))
    scores.append(score)

    # Label: 1 if same speaker, 0 if different speakers
    label = int(speaker1 == speaker2)
    true_labels.append(label)
    predictions.append(int(prediction))

true_labels = np.array(true_labels)
predictions = np.array(predictions)


# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(true_labels, scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()